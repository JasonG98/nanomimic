"""目标窗口、跟踪奖励、软终止与显式状态的可微想象。

单步计分后推进动作时间；sensor 与探索噪声在 checkpoint 重算时复用 RNG 状态。
不改写输入、网络参数或归一化统计量；终止概率保留梯度。
"""

from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from nanomimic.common.rotation import conjugate, inverse_rotate, matrix, multiply, rotate
from nanomimic.common.specs import Motion, RobotSpec
from nanomimic.models.model import Bodies, Dynamics, aux_state, joint_state, represent, world_bodies
from nanomimic.models.policy import Actor, Critic
from nanomimic.rollout.buffer import StartBatch
from nanomimic.rollout.sensor import add_noise, push_history, synth_proprio


def wm_goal(motion: Motion, aux: Tensor, time: Tensor) -> Tensor:
    """十帧目标间隔五控制步，索引钳在所属片段末帧，朝向取旋转矩阵前两列。"""
    indices = motion.window_indices(time)
    quat = aux[:, None, 3:7]
    position = inverse_rotate(quat, motion.body_pos_w[indices, 0] - aux[:, None, :3])
    orientation = matrix(multiply(conjugate(quat), motion.body_quat_w[indices, 0]))[..., :2]
    content = torch.cat((motion.joint_pos[indices], motion.joint_vel[indices]), dim=-1)
    return torch.cat((content.flatten(1), position.flatten(1), orientation.flatten(1)), dim=-1)


def actor_goal(goal: Tensor) -> Tensor:
    """仅移除 10 × 3 的相对位置，保留 content 与 10 × 6 的相对朝向。"""
    return torch.cat((goal[..., :-90], goal[..., -60:]), dim=-1)


def _angle_squared(a: Tensor, b: Tensor) -> Tensor:
    cosine = ((a * b).sum((-2, -1)) - 1) / 2
    return cosine.clamp(-1 + 1e-7, 1 - 1e-7).acos().square()


def reanchor(target: Bodies, aux: Tensor) -> tuple[Tensor, Tensor]:
    """目标根的高度和滚转俯仰保留，xy 与相对偏航按预测根重锚定。"""
    difference = multiply(aux[..., 3:7], conjugate(target.quat[..., 0, :]))
    w, x, y, z = difference.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    turn = torch.stack(((yaw / 2).cos(), zero, zero, (yaw / 2).sin()), dim=-1)
    turn = turn / turn.norm(dim=-1, keepdim=True)
    origin = torch.cat((aux[..., :2], target.pos[..., 0, 2:3]), dim=-1)
    position = origin[..., None, :] + rotate(turn[..., None, :], target.pos - target.pos[..., :1, :])
    orientation = matrix(turn)[..., None, :, :] @ matrix(target.quat)
    return position, orientation


def tracking_reward(
    aux: Tensor, state: Tensor, predicted: Bodies, target: Bodies, action: Tensor, last_action: Tensor, robot: RobotSpec
) -> Tensor:
    """世界系跟踪奖励的正项上界为 5 × dt，关节越限按软限位计算。"""
    target_pos, target_rot = reanchor(target, aux)
    pred_rot = matrix(predicted.quat)
    terms = (
        ((predicted.pos[..., 0, :] - target.pos[..., 0, :]).square().sum(-1), 0.5, 0.3),
        (_angle_squared(pred_rot[..., 0, :, :], matrix(target.quat[..., 0, :])), 0.5, 0.4),
        ((predicted.pos - target_pos).square().sum(-1).mean(-1), 1.0, 0.3),
        (_angle_squared(pred_rot, target_rot).mean(-1), 1.0, 0.4),
        ((predicted.linear - target.linear).square().sum(-1).mean(-1), 1.0, 1.0),
        ((predicted.angular - target.angular).square().sum(-1).mean(-1), 1.0, 3.14),
    )
    positive = torch.stack([weight * (-error / std**2).exp() for error, weight, std in terms]).sum(0)
    position, _ = joint_state(state).chunk(2, dim=-1)
    lo, hi = robot.soft_joint_pos_limits.unbind(-1)
    limit = ((lo - position).clamp_min(0) + (position - hi).clamp_min(0)).sum(-1)
    rate = (action - last_action).square().sum(-1)
    return robot.dt * (positive - 0.1 * rate - 2 * limit)


def termination_probability(aux: Tensor, target: Bodies, robot: RobotSpec) -> Tensor:
    """按锚点高度和投影重力 z 分量误差计算可微终止概率。"""
    index = robot.body_names.index(robot.anchor_body_name)
    height = (aux[..., 2] - target.pos[..., index, 2]).abs()
    gravity_z = 2 * aux[..., 4:6].square().sum(-1) - 1
    target_z = 2 * target.quat[..., index, 1:3].square().sum(-1) - 1
    error = torch.maximum(
        height / robot.anchor_pos_threshold, (gravity_z - target_z).abs() / robot.anchor_ori_threshold
    )
    return ((error - 1) / 0.05).sigmoid()


@dataclass(frozen=True)
class ImagineConfig:
    """robot 为运行时纯张量常量，其余参数由训练 CLI 配置传入。"""

    robot: RobotSpec
    checkpoint: bool = True
    exploration_noise: float = 0.1


@dataclass(frozen=True)
class Rollout:
    reward: Tensor  # [T, B]
    termination: Tensor  # [T, B]
    critic_obs: Tensor  # [T, B, D]
    terminal: Tensor  # [B, D]


def _step(
    dynamics: Dynamics,
    actor: Actor,
    critic: Critic,
    motion: Motion,
    cfg: ImagineConfig,
    state: Tensor,
    aux: Tensor,
    time: Tensor,
    history: Tensor,
    previous: Tensor,
    ctx: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    goal = actor_goal(wm_goal(motion, aux, time))
    observation = critic.observation(state, goal, ctx)
    action = actor(history, goal)
    action = action + cfg.exploration_noise * torch.randn_like(action)
    next_aux, next_state = dynamics(state, action, aux, ctx)
    target = Bodies(
        motion.body_pos_w[time], motion.body_quat_w[time], motion.body_lin_vel_w[time], motion.body_ang_vel_w[time]
    )
    predicted = world_bodies(next_aux, next_state, cfg.robot)
    reward = tracking_reward(next_aux, next_state, predicted, target, action, previous, cfg.robot)
    termination = termination_probability(next_aux, target, cfg.robot)
    next_time = torch.minimum(time + 1, motion.last_frame(time))
    sensor = add_noise(synth_proprio(next_state, next_aux, action, cfg.robot.default_joint_pos))
    return next_state, next_aux, next_time, push_history(history, sensor), action, reward, termination, observation


def imagine(
    dynamics: Dynamics,
    actor: Actor,
    critic: Critic,
    motion: Motion,
    start: StartBatch,
    horizon: int,
    cfg: ImagineConfig,
) -> Rollout:
    if horizon < 1:
        raise ValueError("想象步数必须大于零")
    state, aux = represent(start.full_state), aux_state(start.full_state)
    time, history = start.motion_time, start.sensor
    previous = history[:, -1, -len(cfg.robot.joint_names) :]
    rewards, probabilities, observations = [], [], []
    step = partial(_step, dynamics, actor, critic, motion, cfg)
    for _ in range(horizon):
        args = (state, aux, time, history, previous, start.ctx)
        result = checkpoint(step, *args, use_reentrant=False) if cfg.checkpoint else step(*args)
        state, aux, time, history, previous, reward, probability, observation = result
        rewards.append(reward)
        probabilities.append(probability)
        observations.append(observation)
    terminal = critic.observation(state, actor_goal(wm_goal(motion, aux, time)), start.ctx)
    return Rollout(torch.stack(rewards), torch.stack(probabilities), torch.stack(observations), terminal)


def discounted_return(rollout: Rollout, terminal_value: Tensor, gamma: float, dt: float) -> Tensor:
    """r_t 使用 s_t；bootstrap 使用 s_H。仅钳制价值上界，允许负回报。"""
    survival = torch.cat((torch.ones_like(rollout.reward[:1]), (1 - rollout.termination).cumprod(0)), dim=0)
    discounts = gamma ** torch.arange(rollout.reward.shape[0] + 1, device=rollout.reward.device)
    reward = (discounts[:-1, None] * survival[:-1] * rollout.reward).sum(0)
    bootstrap = terminal_value.clamp_max(5 * dt / (1 - gamma))
    return reward + discounts[-1] * survival[-1] * bootstrap

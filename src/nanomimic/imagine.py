"""显式状态的可微想象 rollout 与存活概率折扣回报。

单步计分后推进动作时间；sensor 与探索噪声在 checkpoint 重算时复用 RNG 状态。
不改写输入、网络参数或归一化统计量；终止概率保留梯度。
"""

from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from nanomimic.buffer import StartBatch
from nanomimic.goals import actor_goal, wm_goal
from nanomimic.model import Bodies, Dynamics, aux_state, represent, world_bodies
from nanomimic.policy import Actor, Critic
from nanomimic.rewards import termination_probability, tracking_reward
from nanomimic.sensor import add_noise, push_history, synth_proprio
from nanomimic.specs import Motion, RobotSpec


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

"""T5 目标、奖励、软终止与想象梯度验收。

数值真值来自参考夹具；开关 checkpoint 使用相同随机种子比较参数梯度。
合成动作只验证数学与梯度，不代替训练效果验收。
"""

import torch
from torch import Tensor

from nanomimic.common.specs import Motion, RobotSpec
from nanomimic.models.model import Bodies, Dynamics, DynamicsConfig, aux_state, represent, world_bodies
from nanomimic.models.policy import Actor, ActorConfig, Critic, CriticConfig
from nanomimic.rollout.buffer import StartBatch
from nanomimic.rollout.imagine import (
    ImagineConfig,
    Rollout,
    actor_goal,
    discounted_return,
    imagine,
    termination_probability,
    tracking_reward,
    wm_goal,
)
from nanomimic.rollout.sensor import synth_proprio


def test_reward_termination_and_goals(reference: dict[str, Tensor], robot: RobotSpec, motion: Motion) -> None:
    def bodies(prefix: str = "") -> Bodies:
        return Bodies(
            *(reference[prefix + name] for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"))
        )

    reward = tracking_reward(
        reference["aux_state"],
        reference["rep_state"],
        bodies(),
        bodies("target_"),
        reference["action"],
        reference["last_action"],
        robot,
    )
    probability = termination_probability(reference["aux_state"], bodies("target_"), robot)
    goal = wm_goal(motion, reference["aux_state"][:6], reference["goal_times"])
    for actual, expected in (
        (reward, reference["reward"]),
        (probability, reference["termination"]),
        (goal, reference["goal"]),
    ):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    assert actor_goal(goal).shape == (6, 640)
    torch.testing.assert_close(actor_goal(goal)[:, :-60], goal[:, :-90])


def test_checkpoint_preserves_actor_gradients(reference: dict[str, Tensor], robot: RobotSpec, motion: Motion) -> None:
    torch.manual_seed(7)
    dynamics = Dynamics(robot, 33, DynamicsConfig((16, 16, 16)))
    actor = Actor(29, ActorConfig(history_length=3, hidden=(16, 12, 8, 8)))
    critic = Critic(29, 33, CriticConfig((16, 8)))
    frame = synth_proprio(
        reference["rep_state"][:4], reference["aux_state"][:4], reference["last_action"][:4], robot.default_joint_pos
    )
    start = StartBatch(
        reference["full_state"][:4],
        torch.tensor([0, 46, 48, 95]),
        torch.zeros(4, 33),
        frame[:, None].expand(-1, 3, -1),
        torch.zeros(4, 3, 29),
        torch.ones(4, 3, dtype=torch.bool),
    )
    gradients, rewards = [], []
    counts = [normalizer.count.clone() for normalizer in (actor.obs_norm, actor.goal_norm, dynamics.state_norm)]
    for enabled in (False, True):
        actor.zero_grad(set_to_none=True)
        torch.manual_seed(19)
        rollout = imagine(dynamics, actor, critic, motion, start, 4, ImagineConfig(robot, checkpoint=enabled))
        loss = -discounted_return(rollout, critic(rollout.terminal), 0.99, robot.dt).mean()
        loss.backward()
        parts = []
        for parameter in actor.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            parts.append(parameter.grad.flatten())
        gradients.append(torch.cat(parts))
        rewards.append(rollout.reward.detach())
        assert gradients[-1].abs().sum() > 0
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(rewards[0], rewards[1], atol=0, rtol=0)
    for before, normalizer in zip(counts, (actor.obs_norm, actor.goal_norm, dynamics.state_norm), strict=True):
        assert torch.equal(before, normalizer.count)


def test_survival_return_and_bootstrap() -> None:
    rollout = Rollout(
        torch.tensor([[2.0], [3.0], [5.0]]),
        torch.tensor([[0.2], [0.5], [1.0]]),
        torch.empty(3, 1, 0),
        torch.empty(1, 0),
    )
    actual = discounted_return(rollout, torch.tensor([100.0]), 0.9, 0.02)
    torch.testing.assert_close(actual, torch.tensor([2 + 0.9 * 0.8 * 3 + 0.9**2 * 0.4 * 5]))
    zero = Rollout(torch.zeros(2, 1), torch.zeros(2, 1), torch.empty(2, 1, 0), torch.empty(1, 0))
    torch.testing.assert_close(discounted_return(zero, torch.tensor([100.0]), 0.9, 0.02), torch.tensor([0.81]))


def test_default_network_contract() -> None:
    assert ActorConfig().history_length == 1
    assert ActorConfig().hidden == (2048, 1024, 512, 256)
    assert ActorConfig().latent_dim == 32
    assert CriticConfig().hidden == (256, 128)


def test_step_scores_before_advancing_clip(reference: dict[str, Tensor], robot: RobotSpec, motion: Motion) -> None:
    torch.manual_seed(5)
    dynamics = Dynamics(robot, 33, DynamicsConfig((8, 8)))
    actor = Actor(29, ActorConfig(hidden=(8, 8)))
    critic = Critic(29, 33, CriticConfig((8,)))
    full = reference["full_state"][:2]
    aux, state = aux_state(full), represent(full)
    last = torch.zeros(2, 29)
    sensor = synth_proprio(state, aux, last, robot.default_joint_pos)[:, None]
    time = torch.tensor([46, 95])
    ctx = torch.zeros(2, 33)
    start = StartBatch(full, time, ctx, sensor, last[:, None], torch.ones(2, 1, dtype=torch.bool))
    rollout = imagine(dynamics, actor, critic, motion, start, 1, ImagineConfig(robot, False, 0))
    goal = actor_goal(wm_goal(motion, aux, time))
    action = actor(sensor, goal)
    next_aux, next_state = dynamics(state, action, aux, ctx)
    target = Bodies(
        motion.body_pos_w[time], motion.body_quat_w[time], motion.body_lin_vel_w[time], motion.body_ang_vel_w[time]
    )
    expected = tracking_reward(
        next_aux, next_state, world_bodies(next_aux, next_state, robot), target, action, last, robot
    )
    torch.testing.assert_close(rollout.reward[0], expected)
    terminal_goal = actor_goal(wm_goal(motion, next_aux, torch.tensor([47, 95])))
    torch.testing.assert_close(rollout.terminal, critic.observation(next_state, terminal_goal, ctx))
    torch.testing.assert_close(rollout.critic_obs[0], critic.observation(state, goal, ctx))


def test_soft_threshold_and_identical_pose_gradients(reference: dict[str, Tensor], robot: RobotSpec) -> None:
    aux = reference["aux_state"].clone().requires_grad_()
    target = Bodies(
        reference["body_pos_w"], reference["body_quat_w"], reference["body_lin_vel_w"], reference["body_ang_vel_w"]
    )
    changed = aux.detach().clone()
    changed[:, 2] += robot.anchor_pos_threshold
    torch.testing.assert_close(
        termination_probability(changed, target, robot), torch.full((32,), 0.5), atol=2e-6, rtol=0
    )
    state = reference["rep_state"]
    predicted = world_bodies(aux, state, robot)
    reward = tracking_reward(aux, state, predicted, target, reference["action"], reference["action"], robot)
    (gradient,) = torch.autograd.grad(reward.sum(), aux)
    assert torch.isfinite(gradient).all()

"""T6 合成数据学习、优化器分组与 GAE 验收。

监督轨迹来自已对拍的解析积分，不用世界模型自身输出作标签。
CPU 网络显式缩小；保留真实 Muon / Adam、裁剪与反向传播。
"""

from dataclasses import replace

import pytest
import torch
from torch import nn

from nanomimic.common.specs import Motion, RobotSpec
from nanomimic.models.model import Dynamics, DynamicsConfig, aux_state, integrate, reconstruct_full_state, represent
from nanomimic.models.policy import Actor, ActorConfig, Critic, CriticConfig
from nanomimic.rollout.buffer import ReplayBatch, ReplayBuffer
from nanomimic.training.config import PolicyTrainConfig, TrainConfig, WorldModelConfig
from nanomimic.training.learning import PolicyImprover, WorldModelFitter, gae_returns
from nanomimic.training.optim import Optimizers


def synthetic_batch(robot: RobotSpec, batch: int = 16, horizon: int = 3) -> ReplayBatch:
    """动作驱动小幅速度增量的确定性系统，首状态随机扰动。"""
    generator = torch.Generator().manual_seed(23)
    full = torch.zeros(batch, 71)
    full[:, 2], full[:, 3] = 0.8, 1
    full[:, 7:36] = robot.default_joint_pos + 0.01 * torch.randn(batch, 29, generator=generator)
    aux, state = aux_state(full), represent(full)
    actions = 0.1 * torch.randn(batch, horizon + 1, 29, generator=generator)
    states = [full]
    for time in range(horizon):
        action = actions[:, time]
        delta = torch.cat((0.01 * action[:, :6], 0.02 * action), dim=-1)
        aux, state = integrate(aux, state, delta, robot.dt)
        states.append(reconstruct_full_state(aux, state))
    return ReplayBatch(
        torch.stack(states, dim=1),
        torch.zeros(batch, horizon + 1, 93),
        actions,
        torch.arange(horizon + 1).expand(batch, -1),
        torch.zeros(batch, horizon + 1, dtype=torch.bool),
        torch.zeros(batch, horizon + 1, dtype=torch.bool),
        torch.zeros(batch, horizon + 1, 33),
    )


def test_world_model_loss_decreases(robot: RobotSpec) -> None:
    torch.manual_seed(31)
    dynamics = Dynamics(robot, 33, DynamicsConfig((32, 32, 32)))
    cfg = TrainConfig(compile=False, wm=WorldModelConfig(lr=2e-3, batch=16, horizon=3, num_batches=1))
    fitter, batch = WorldModelFitter(dynamics, cfg), synthetic_batch(robot)
    dynamics.state_norm.update(represent(batch.full_state))
    dynamics.action_norm.update(dynamics.process_action(batch.action))
    initial = float(fitter.evaluate(batch).detach())
    for _ in range(60):
        assert torch.isfinite(torch.tensor(fitter.update(batch)))
    final = float(fitter.evaluate(batch).detach())
    print(f"\n合成数据世界模型损失：{initial:.8g} → {final:.8g}")
    assert final < initial * 0.5


def test_optimizer_partition_and_clipping() -> None:
    actor = Actor(29, ActorConfig(hidden=(8, 8, 8, 8)))
    optim = Optimizers(actor, 2e-4, beta2=0.95)
    adam, muon = optim.optimizers
    assert isinstance(adam, torch.optim.Adam) and isinstance(muon, torch.optim.Muon)
    expected = set()
    for network in (actor.encoder, actor.decoder):
        layers = [module for module in network if isinstance(module, nn.Linear)]
        expected.update(id(layer.weight) for layer in layers[1:-1])
    actual = {id(p) for group in muon.param_groups for p in group["params"]}
    other = {id(p) for group in adam.param_groups for p in group["params"]}
    assert actual == expected and not actual & other
    assert actual | other == {id(p) for p in actor.parameters()}
    assert adam.param_groups[0]["betas"] == (0.9, 0.95)
    loss = torch.stack([(parameter * 100).square().sum() for parameter in actor.parameters()]).sum()
    optim.step(loss, element_clip=0.1)
    assert all(parameter.grad is not None and parameter.grad.abs().max() <= 0.1 for parameter in actor.parameters())
    restored = Optimizers(actor, 1e-3, beta2=0.95)
    restored.load_state_dict(optim.state_dict())
    assert all(group["lr"] == 1e-3 for optimizer in restored.optimizers for group in optimizer.param_groups)
    assert restored.optimizers[0].state
    with pytest.raises(FloatingPointError, match="非有限"):
        optim.step(torch.tensor(float("nan")))


def test_gae_handles_soft_termination() -> None:
    values = torch.tensor([[1.0], [2.0], [3.0]])
    rewards = torch.tensor([[0.5], [1.5], [2.5]])
    probabilities = torch.tensor([[0.2], [0.5], [1.0]])
    actual = gae_returns(values, torch.tensor([99.0]), rewards, probabilities, 0.9, 0.95)
    a2 = 2.5 - 3
    a1 = 1.5 + 0.9 * 0.5 * 3 - 2 + 0.9 * 0.95 * 0.5 * a2
    a0 = 0.5 + 0.9 * 0.8 * 2 - 1 + 0.9 * 0.95 * 0.8 * a1
    torch.testing.assert_close(actual, torch.tensor([[a0 + 1], [a1 + 2], [a2 + 3]]))


def test_compiled_world_model_matches_eager(robot: RobotSpec) -> None:
    torch.manual_seed(4)
    eager = Dynamics(robot, 33, DynamicsConfig((8, 8)))
    compiled = Dynamics(robot, 33, DynamicsConfig((8, 8)))
    compiled.load_state_dict(eager.state_dict())
    cfg = TrainConfig(compile=False)
    plain = WorldModelFitter(eager, cfg)
    traced = WorldModelFitter(compiled, replace(cfg, compile=True, compile_backend="aot_eager"))
    batch = synthetic_batch(robot, batch=2, horizon=2)
    losses = [fitter.evaluate(batch) for fitter in (plain, traced)]
    for loss in losses:
        loss.backward()
    torch.testing.assert_close(losses[0], losses[1])
    for a, b in zip(eager.parameters(), compiled.parameters(), strict=True):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("horizon", [2, 4])
def test_policy_compile_and_frozen_dynamics(robot: RobotSpec, motion: Motion, horizon: int) -> None:
    gradients = []
    for compiled in (False, True):
        torch.manual_seed(15)
        dynamics = Dynamics(robot, 33, DynamicsConfig((8, 8)))
        actor = Actor(29, ActorConfig(hidden=(8, 8)))
        critic = Critic(29, 33, CriticConfig((8, 8)))
        cfg = TrainConfig(
            compile=compiled, compile_backend="aot_eager", policy=PolicyTrainConfig(batch=2, horizon=horizon)
        )
        improver = PolicyImprover(dynamics, actor, critic, motion, robot, cfg)
        batch = synthetic_batch(robot, batch=2, horizon=2)
        buffer = ReplayBuffer(2, 8)
        for time in range(3):
            buffer.append(
                ReplayBatch(
                    batch.full_state[:, time],
                    batch.sensor[:, time],
                    batch.action[:, time],
                    batch.motion_time[:, time],
                    batch.terminated[:, time],
                    batch.truncated[:, time],
                    batch.ctx[:, time],
                )
            )
        before = {name: value.clone() for name, value in dynamics.state_dict().items()}
        torch.manual_seed(90)
        metrics = improver.improve(buffer)
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        for name, value in dynamics.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        assert all(parameter.requires_grad and parameter.grad is None for parameter in dynamics.parameters())
        parts = [parameter.grad.flatten() for parameter in actor.parameters() if parameter.grad is not None]
        gradients.append(torch.cat(parts))
        assert torch.isfinite(gradients[-1]).all() and gradients[-1].abs().sum() > 0
        # 每次策略改进确实执行 5×4 个价值网络 minibatch。
        adam = improver.critic_optim.optimizers[0]
        assert all(int(state["step"]) == 20 for state in adam.state.values())
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-6, rtol=1e-5)

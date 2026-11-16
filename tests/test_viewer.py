"""T9 的真实 CPU 回放、预测闭环与对齐语义验收。

使用发布版仿真与已有动作夹具，虚影的预期状态由独立前向调用计算。
原生绘制通过真实 MuJoCo 场景核对；不以试跑策略表现作为效果验收。
"""

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from mjlab.viewer.debug_visualizer import DebugVisualizer, NullDebugVisualizer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
from mujoco import MjvScene  # pyright: ignore[reportAttributeAccessIssue]
from torch import Tensor, nn

from nanomimic.cli import PlayCmd, parse_command
from nanomimic.common.specs import SimConfig
from nanomimic.models.model import Dynamics, DynamicsConfig, reconstruct_full_state
from nanomimic.models.policy import Actor, ActorConfig
from nanomimic.rollout.imagine import actor_goal, wm_goal
from nanomimic.runtime.sim import Sim
from nanomimic.runtime.viewer import GhostRollout, NativeGhostViewer


@pytest.fixture(scope="module")
def sim() -> Iterator[Sim]:
    with Sim(SimConfig(Path(__file__).parent / "fixtures/motion.npz", num_envs=2, device="cpu")) as value:
        yield value


@pytest.fixture
def ghost(sim: Sim) -> Iterator[GhostRollout]:
    torch.manual_seed(12)
    actor = Actor(29, ActorConfig(history_length=3, hidden=(16, 12)))
    dynamics = Dynamics(sim.robot, 33, DynamicsConfig((16, 16)))
    with torch.no_grad():
        # 固定为零速度增量，避免未训练动力学影响重置边界的测试。
        output = cast(nn.Linear, dynamics.mlp[-1])
        output.weight.zero_()
        output.bias.zero_()
    value = GhostRollout(sim, actor, dynamics, num_steps_rollout=2)
    try:
        yield value
    finally:
        value.close()


def test_prediction_uses_its_own_sensor_goal_and_action(ghost: GhostRollout) -> None:
    ghost.num_steps_rollout = 100
    ghost.history.push(ghost.history.window()[:, -1] + 0.4)
    predicted_history = ghost.history.window().clone()
    predicted_goal = actor_goal(wm_goal(ghost.sim.motion, ghost.aux, ghost.time))
    before = {name: value.clone() for name, value in ghost.dynamics.state_dict().items()}
    with torch.no_grad():
        predicted_action = ghost.actor(predicted_history, predicted_goal)
        expected_aux, expected_state = ghost.dynamics(ghost.state, predicted_action, ghost.aux, ghost.ctx)
    calls: list[tuple[Tensor, Tensor]] = []
    applied: list[Tensor] = []

    def record(_module: nn.Module, args: tuple[Tensor, ...], _output: Tensor) -> None:
        calls.append((args[0].clone(), args[1].clone()))

    def record_dynamics(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        applied.append(args[1].clone())

    handle = ghost.actor.register_forward_hook(record)
    dynamics_handle = ghost.dynamics.register_forward_pre_hook(record_dynamics)
    try:
        done = ghost.step()
    finally:
        handle.remove()
        dynamics_handle.remove()
    assert not done[0] and len(calls) == 2
    assert calls[0][0].shape == (2, 3, 93) and calls[1][0].shape == (1, 3, 93)
    assert not torch.equal(calls[0][0][:1], calls[1][0])
    torch.testing.assert_close(calls[1][0], predicted_history, atol=0, rtol=0)
    torch.testing.assert_close(calls[1][1], predicted_goal, atol=0, rtol=0)
    assert len(applied) == 1
    torch.testing.assert_close(applied[0], predicted_action, atol=0, rtol=0)
    torch.testing.assert_close(ghost.aux, expected_aux, atol=1e-5, rtol=0)
    torch.testing.assert_close(ghost.state, expected_state, atol=1e-5, rtol=0)
    torch.testing.assert_close(ghost.history.window()[:, -1, -29:], predicted_action, atol=0, rtol=0)
    assert not ghost.state.requires_grad and ghost.state.shape[0] == 1
    for name, value in ghost.dynamics.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def _assert_aligned(ghost: GhostRollout) -> None:
    torch.testing.assert_close(
        reconstruct_full_state(ghost.aux, ghost.state), ghost.obs.full_state[:1], atol=1e-5, rtol=0
    )
    assert ghost.steps == 0 and torch.equal(ghost.time, ghost.sim.motion_time[:1])
    torch.testing.assert_close(ghost.history.window()[:, 0], ghost.history.window()[:, -1], atol=0, rtol=0)
    np.testing.assert_allclose(
        ghost.qpos[ghost.joint_indices], ghost.obs.full_state[0, 7:36].numpy(), atol=1e-5, rtol=0
    )


def test_periodic_alignment_keeps_real_history(ghost: GhostRollout) -> None:
    initial = ghost.alignments
    ghost.step()
    previous_frame = ghost.real_history.window()[:, -1].clone()
    ghost.step()
    assert ghost.alignments == initial + 1 and ghost.periodic_alignments == 1
    _assert_aligned(ghost)
    torch.testing.assert_close(ghost.real_history.window()[:, -2], previous_frame, atol=0, rtol=0)


def test_other_environment_done_does_not_realign_env_zero(ghost: GhostRollout) -> None:
    ghost.num_steps_rollout = 100
    initial = ghost.alignments
    ghost.sim._env.episode_length_buf[1] = ghost.sim._env.max_episode_length - 1
    done = ghost.step()
    assert done.tolist() == [False, True] and ghost.alignments == initial
    history = ghost.real_history.window()
    torch.testing.assert_close(history[1, 0], history[1, -1], atol=0, rtol=0)


def test_episode_end_and_manual_reset_realign(ghost: GhostRollout) -> None:
    ghost.sim._env.episode_length_buf[0] = ghost.sim._env.max_episode_length - 1
    ghost._episode_step = ghost.sim._env.max_episode_length - 1
    assert ghost.step()[0] and ghost.done_alignments == 1
    _assert_aligned(ghost)
    viewer = NativeGhostViewer(ghost)
    initial = ghost.alignments
    viewer.reset_environment()
    assert ghost.alignments == initial + 1
    _assert_aligned(ghost)
    # 模拟 Viser 控件直接重置底层环境，下一步必须识别并刷新两侧历史。
    ghost.step()
    initial = ghost.alignments
    ghost.sim.reset()
    ghost.step()
    assert ghost.alignments == initial + 1


def test_real_native_ghost_drawing_and_model_isolation(ghost: GhostRollout) -> None:
    original = ghost.sim._env.sim.mj_model
    assert ghost.model is not original
    np.testing.assert_allclose(ghost.model.geom_rgba, np.tile([0.5, 0.5, 0.8, 0.8], (original.ngeom, 1)), atol=1e-7)
    assert not np.array_equal(original.geom_rgba, ghost.model.geom_rgba)
    scene = MjvScene(original, maxgeom=10000)
    visualizer = MujocoNativeDebugVisualizer(scene, original, env_idx=0)
    ghost.draw(visualizer)
    assert scene.ngeom > 0
    colors = np.stack([geom.rgba for geom in scene.geoms[: scene.ngeom]])
    # 发布版 native ghost 接口打开 mjVIS_TRANSPARENT，按模型 map.alpha 再乘透明度。
    expected = [0.5, 0.5, 0.8, 0.8 * ghost.model.vis.map.alpha]
    assert np.any(np.all(np.isclose(colors, expected, atol=1e-7), axis=-1))
    ghost.draw(cast(DebugVisualizer, NullDebugVisualizer(env_idx=1)))


def test_play_cli_and_invalid_rollout(sim: Sim) -> None:
    command = parse_command(["play", "--checkpoint", "runs/100", "--viewer", "viser", "--num-steps-rollout", "5"])
    assert isinstance(command, PlayCmd) and command.viewer == "viser" and command.num_steps_rollout == 5
    assert PlayCmd(Path("runs/100")).num_steps_rollout == 1_000_000
    with pytest.raises(ValueError, match="num_steps_rollout"):
        GhostRollout(sim, Actor(29, ActorConfig(hidden=(8,))), Dynamics(sim.robot, 33, DynamicsConfig((8,))), 0)

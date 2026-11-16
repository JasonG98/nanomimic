"""发布版 mjlab 的 CPU 集成验收。

使用真实 G1 仿真与生成器产出的本地合成动作，不使用仿真器替身。
覆盖随机动作、自动重置、上下文与片段边界；不宣称 GPU 或训练效果验收。
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from nanomimic.sim import Sim
from nanomimic.specs import SimConfig


@pytest.fixture(scope="module")
def sim() -> Iterator[Sim]:
    with Sim(SimConfig(Path(__file__).parent / "fixtures/motion.npz", num_envs=2, device="cpu")) as env:
        yield env


def test_registered_task_and_robot_constants(sim: Sim) -> None:
    cfg = load_env_cfg("Nanomimic-Tracking-Flat-Unitree-G1")
    assert cfg.scene.num_envs == 2048, "注册配置不能被首次创建的 CPU 实例改写"
    assert set(cfg.observations) == {"actor", "goal", "full_state"}
    assert set(cfg.events) == {"foot_friction", "base_com", "encoder_bias"}
    assert "self_collisions" not in cfg.rewards
    assert "ee_body_pos" not in cfg.terminations
    assert cfg.rewards["joint_limit"].weight == -2
    assert all(not group.enable_corruption for group in cfg.observations.values())
    assert sim._command.cfg.sampling_mode == "adaptive"
    robot = sim.robot
    assert len(robot.joint_names) == 29
    assert len(robot.body_names) == 14
    assert robot.anchor_body_name == "pelvis"
    assert robot.dt == 0.02
    assert robot.action_scale.shape == robot.action_offset.shape == (29,)
    assert robot.soft_joint_pos_limits.shape == (29, 2)
    assert robot.body_pos.shape == (30, 3)
    assert robot.body_quat.shape == (30, 4)
    assert robot.joint_axis.shape == robot.joint_anchor.shape == (29, 3)
    assert robot.parents[0] == robot.body_joint[0] == -1
    assert all(0 <= parent < child for child, parent in enumerate(robot.parents[1:], 1))
    assert sorted(col for col in robot.body_joint if col >= 0) == list(range(29))
    assert robot.anchor_pos_threshold == 0.25
    assert robot.anchor_ori_threshold == 0.8
    torch.testing.assert_close(robot.action_offset, robot.default_joint_pos)
    with np.load(Path(__file__).parent / "fixtures/reference.npz") as ref:
        for field in (
            "parents",
            "body_joint",
            "body_pos",
            "body_quat",
            "joint_axis",
            "joint_anchor",
            "joint_reference",
            "tracked_body_indices",
        ):
            torch.testing.assert_close(
                torch.as_tensor(getattr(robot, field)),
                torch.from_numpy(ref[f"robot_{field}"]),
                atol=1e-5,
                rtol=0,
                check_dtype=False,
            )
        torch.testing.assert_close(
            robot.soft_joint_pos_limits, torch.from_numpy(ref["soft_joint_pos_limits"]), atol=1e-5, rtol=0
        )
    assert sim.motion.body_pos_w.shape == (96, 14, 3)
    assert sim.motion.body_quat_w.shape == (96, 14, 4)
    assert sim.motion.joint_pos.shape == sim.motion.joint_vel.shape == (96, 29)
    assert sim.motion.clip_ends.tolist() == [47, 95]


def test_random_actions_100_steps(sim: Sim) -> None:
    previous = sim.reset()
    saved = previous.full_state.clone()
    generator = torch.Generator().manual_seed(17)
    done_count = 0
    for _ in range(100):
        action = torch.rand(2, 29, generator=generator) * 2 - 1
        obs, reward, terminated, truncated = sim.step(action)
        assert obs.full_state.shape == (2, 71)
        assert obs.goal.shape == (2, 670)
        assert obs.sensor.shape == (2, 93)
        assert reward.shape == terminated.shape == truncated.shape == (2,)
        assert terminated.dtype == truncated.dtype == torch.bool
        for value in (obs.full_state, obs.goal, obs.sensor, reward):
            assert torch.isfinite(value).all()
        done = terminated | truncated
        done_count += int(done.sum())
        torch.testing.assert_close(obs.sensor[~done, -29:], action[~done], atol=0, rtol=0)
        assert obs.sensor[done, -29:].eq(0).all()
        assert sim.motion_time.ge(0).all() and sim.motion_time.lt(96).all()
    assert done_count > 0, "随机动作必须实际覆盖自动重置路径"
    torch.testing.assert_close(previous.full_state, saved, atol=0, rtol=0)


def test_context_reads_physical_parameters(sim: Sim) -> None:
    ctx = sim.read_ctx()
    assert sim.ctx_spec.names == ("foot_friction", "base_com", "encoder_bias")
    assert sim.ctx_spec.widths == (1, 3, 29)
    assert ctx.shape == (2, 33)
    assert torch.isfinite(ctx).all() and ctx.abs().max() <= 1 + 1e-6
    physical = (ctx + 1) / 2 * (sim.ctx_spec.upper - sim.ctx_spec.lower) + sim.ctx_spec.lower
    model = sim._env.sim.model
    torch.testing.assert_close(physical[:, 0], model.geom_friction[:, sim._foot_id, 0])
    torch.testing.assert_close(physical[:, 1:4], model.body_ipos[:, sim._com_id] - sim._nominal_com)
    torch.testing.assert_close(physical[:, 4:], sim._env.scene["robot"].data.encoder_bias)
    assert not torch.equal(ctx[0], ctx[1]), "随机化必须在环境之间生效"


def test_observations_match_reference(sim: Sim) -> None:
    env = sim._env
    robot = env.scene["robot"]
    sim.reset()
    with np.load(Path(__file__).parent / "fixtures/reference.npz") as ref:
        for start in range(0, 6, 2):
            state = torch.from_numpy(ref["full_state"][start : start + 2]).clone()
            root = torch.cat((state[:, :7], state[:, 36:42]), dim=-1)
            root[:, :3] += env.scene.env_origins
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(state[:, 7:36], state[:, 42:])
            sim._command.time_steps[:] = torch.from_numpy(ref["goal_times"][start : start + 2])
            env.sim.forward()
            env.sim.sense()
            obs = sim._snapshot(env.observation_manager.compute(update_history=True))
            torch.testing.assert_close(obs.full_state, state, atol=1e-5, rtol=0)
            torch.testing.assert_close(obs.goal, torch.from_numpy(ref["goal"][start : start + 2]), atol=1e-5, rtol=0)
            torch.testing.assert_close(
                obs.sensor[:, :3], torch.from_numpy(ref["rep_state"][start : start + 2, 1:4]), atol=1e-5, rtol=0
            )
            torch.testing.assert_close(
                obs.sensor[:, 6:35],
                state[:, 7:36] - sim.robot.default_joint_pos + robot.data.encoder_bias,
                atol=1e-5,
                rtol=0,
            )
            repeated = sim._snapshot(env.observation_manager.compute(update_history=True))
            torch.testing.assert_close(obs.sensor, repeated.sensor, atol=0, rtol=0)
    sim.reset()


def test_clip_end_does_not_teleport_or_cross_clip(sim: Sim) -> None:
    sim.reset()
    sim._command.time_steps[:] = torch.tensor([47, 95])
    before = sim._env.scene["robot"].data.root_link_pos_w.clone()
    sim._command._update_command()
    assert sim.motion_time.tolist() == [47, 95]
    torch.testing.assert_close(sim._env.scene["robot"].data.root_link_pos_w, before, atol=0, rtol=0)
    sim.reset()


def test_rejects_invalid_actions(sim: Sim) -> None:
    with pytest.raises(ValueError, match="action"):
        sim.step(torch.zeros(2, 28))
    with pytest.raises(ValueError, match="有限值"):
        sim.step(torch.full((2, 29), float("nan")))


def test_time_limit_returns_new_episode_observation(sim: Sim) -> None:
    sim.reset()
    sim._env.episode_length_buf.fill_(sim._env.max_episode_length - 1)
    obs, _, _, truncated = sim.step(torch.full((2, 29), 0.1))
    assert truncated.all()
    assert sim._env.episode_length_buf.eq(0).all()
    assert obs.sensor[:, -29:].eq(0).all()

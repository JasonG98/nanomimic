"""完整动作评估的首次尝试统计与真实 CPU 边界验收。

失败后的重置、短回合与超时不能增加完整动作成功率。
误差包含失败状态，评估不修改策略参数或归一化统计量。
"""

import json
from dataclasses import replace
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
import torch

from nanomimic.common.specs import SimConfig
from nanomimic.models.policy import Actor, ActorConfig
from nanomimic.runtime.export import export
from nanomimic.runtime.sim import Sim
from nanomimic.training.config import TrainConfig, config_dict

evaluation = run_path(str(Path(__file__).parents[1] / "tools/evaluate_tracking.py"))
TrackingScore = evaluation["TrackingScore"]
_reset_at_start = evaluation["_reset_at_start"]
evaluate_policy = evaluation["evaluate_policy"]


def test_first_attempt_success_and_terminal_errors() -> None:
    score = TrackingScore(3, ("error_body_pos",), torch.device("cpu"))
    # 第一行在首步摔倒，第二行完整走三步，第三行在末步超时。
    score.add(torch.tensor([[3.0], [1.0], [2.0]]), torch.tensor([True, False, False]), torch.zeros(3, dtype=torch.bool))
    score.add(torch.tensor([[999.0], [2.0], [4.0]]), torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))
    score.add(
        torch.tensor([[float("nan")], [3.0], [6.0]]),
        torch.zeros(3, dtype=torch.bool),
        torch.tensor([False, False, True]),
    )
    result = score.summary(3, 0.02)
    assert result["successes"] == 1
    assert result["terminated_trials"] == result["truncated_trials"] == 1
    assert result["errors_all_attempted_steps"]["error_body_pos"] == pytest.approx(21 / 7)
    assert result["errors_successful_trials"]["error_body_pos"] == 2
    assert [trial["steps"] for trial in result["per_trial"]] == [1, 3, 3]
    assert result["mean_survival_seconds"] == pytest.approx(7 / 3 * 0.02)


def test_short_survival_and_nonfinite_errors_cannot_pass() -> None:
    score = TrackingScore(1, ("error_body_pos",), torch.device("cpu"))
    with pytest.raises(ValueError, match="非有限"):
        score.add(torch.tensor([[float("nan")]]), torch.tensor([False]), torch.tensor([False]))
    score.add(torch.tensor([[0.1]]), torch.tensor([False]), torch.tensor([False]))
    with pytest.raises(ValueError, match="尚未走完"):
        score.summary(100, 0.02)
    score.add(torch.tensor([[0.2]]), torch.tensor([True]), torch.tensor([False]))
    result = score.summary(100, 0.02)
    assert result["success_rate"] == 0
    assert result["errors_successful_trials"] is None
    assert result["mean_completed_fraction"] == pytest.approx(0.02)


def test_real_cpu_full_clip_protocol_and_frozen_actor(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures/motion.npz"
    with np.load(source) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    data["motion_ends"] = np.array([data["joint_pos"].shape[0]])
    motion_file = tmp_path / "single.npz"
    np.savez(motion_file, **data)
    cfg = SimConfig(motion_file, num_envs=2, device="cpu", seed=7)
    actor_cfg = ActorConfig(history_length=3, hidden=(16, 12, 8, 8))
    with Sim(cfg) as sim:
        actor = Actor(29, actor_cfg)
        original = {key: value.clone() for key, value in actor.state_dict().items()}
        result = evaluate_policy(sim, actor)
        assert result["trials"] == 2
        assert result["expected_steps"] == 95
        assert result["successes"] + result["terminated_trials"] + result["truncated_trials"] == 2
        assert result["truncated_trials"] == 0
        assert not sim._env.cfg.auto_reset
        assert sim._env.max_episode_length > 96
        assert set(sim._env.cfg.events) == {"foot_friction", "base_com", "encoder_bias"}
        torch.testing.assert_close(actor.state_dict(), original, rtol=0, atol=0)
        obs = _reset_at_start(sim)
        assert sim.motion_time.eq(0).all()
        torch.testing.assert_close(obs.full_state[:, :3], sim.motion.body_pos_w[0, 0].expand(2, -1), rtol=0, atol=1e-6)
        # 一行局部重置不能改变另一行的状态或动作帧。
        sim.step(torch.zeros(2, 29))
        before = sim.motion_time.clone()
        obs = _reset_at_start(sim, torch.tensor([0]))
        assert sim.motion_time.tolist() == [0, int(before[1])]
        assert torch.isfinite(obs.full_state).all()
    with Sim(replace(cfg, motion_file=source)) as sim, pytest.raises(ValueError, match="单片段"):
        evaluate_policy(sim, actor)
    # 保存独立推理检查点，通过旧路径调用入口，默认产物必须归入新子目录。
    checkpoint = tmp_path / "run/checkpoints/00000007"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(config_dict(TrainConfig(motion_file=motion_file, actor=actor_cfg)))
    )
    torch.save({"schema_version": 1, "iteration": 7, "model": actor.state_dict()}, checkpoint / "actor.pt")
    old_path = checkpoint.parent.parent / checkpoint.name
    result_path = evaluation["evaluate"](evaluation["Config"](checkpoint=old_path, num_envs=2, device="cpu"))
    assert result_path == tmp_path / "run/evaluations/eval-00000007-seed0.json"
    assert json.loads(result_path.read_text())["checkpoint"] == str(checkpoint)
    export(old_path, verify=True)
    assert (tmp_path / "run/exports/actor-00000007.onnx").is_file()

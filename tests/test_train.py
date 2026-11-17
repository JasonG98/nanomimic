"""T6 真实 CPU 训练与检查点恢复验收。

使用发布版 mjlab、两环境和显式缩小的网络，检查真实采集到训练的闭环。
恢复保留参数及优化器，回放重新预热；不声称 GPU 或收敛效果验证。
"""

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import nanomimic.training.train as training
from nanomimic.cli import TrainCmd, parse_command
from nanomimic.common.specs import SimConfig
from nanomimic.models.model import DynamicsConfig
from nanomimic.models.policy import ActorConfig, CriticConfig
from nanomimic.runtime.sim import Sim
from nanomimic.training.config import PolicyTrainConfig, TrainConfig, WorldModelConfig, load_config, resolve_config
from nanomimic.training.train import ScalarSink, Trainer


def _small_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        motion_file=Path(__file__).parent / "fixtures/motion.npz",
        num_envs=2,
        device="cpu",
        max_iterations=4,
        buffer_capacity=16,
        replay_ratio=1,
        compile=False,
        output=tmp_path / "initial",
        log_interval=2,
        save_interval=2,
        dynamics=DynamicsConfig((16, 16, 16)),
        actor=ActorConfig(history_length=3, hidden=(16, 12, 8, 8)),
        critic=CriticConfig((16, 8)),
        wm=WorldModelConfig(batch=2, horizon=2, num_batches=1),
        policy=PolicyTrainConfig(batch=2, horizon=2),
    )


def test_real_training_schedule_and_resume(tmp_path: Path) -> None:
    cfg = resolve_config(_small_config(tmp_path))
    assert cfg.motion_file is not None
    with Sim(SimConfig(cfg.motion_file, cfg.num_envs, cfg.device, cfg.seed)) as sim:
        trainer = Trainer(sim, cfg)
        initial = {name: next(model.parameters()).detach().clone() for name, (model, _) in trainer.models().items()}
        metrics = []
        for _ in range(4):
            metrics.append(trainer.step())
        assert [metric["updates"] for metric in metrics] == [0, 0, 1, 2]
        assert trainer.credit == 0
        assert all(torch.isfinite(torch.tensor(list(metric.values()))).all() for metric in metrics)
        for name, (model, _) in trainer.models().items():
            assert not torch.equal(initial[name], next(model.parameters()))
        assert trainer.actor.obs_norm.count == trainer.actor.goal_norm.count == 8
        assert trainer.dynamics.state_norm.count == trainer.dynamics.action_norm.count == 8
        # 历史与 buffer 复用同一份噪声，动作槽与下一观测的上一动作对齐。
        data = trainer.buffer._data
        alive = ~(data["terminated"][:, :3] | data["truncated"][:, :3])
        torch.testing.assert_close(data["sensor"][:, 1:4, -29:][alive], data["action"][:, :3][alive])
        saved = trainer.save()
        assert {path.name for path in saved.iterdir()} == {"world_model.pt", "actor.pt", "critic.pt", "config.json"}
        assert load_config(saved).actor == cfg.actor
        assert json.loads((saved / "config.json").read_text())["schema_version"] == 1
        command = parse_command(
            [
                "train",
                "--resume",
                str(trainer.output / saved.name),
                "--max-iterations",
                "8",
                "--output",
                str(tmp_path / "resumed"),
                "--log-interval",
                "10",
            ]
        )
        assert isinstance(command, TrainCmd) and command.log_interval == 10
        resume_cfg = resolve_config(command)
        assert resume_cfg.resume == saved
        resumed = Trainer(sim, resume_cfg)
        assert resumed.iteration == 4 and resumed.updates == 2 and resumed.buffer.size == 0
        for name, (model, optim) in trainer.models().items():
            restored_model, restored_optim = resumed.models()[name]
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored_model.state_dict()[key], atol=0, rtol=0)
            for before, after in zip(optim.state_dict(), restored_optim.state_dict(), strict=True):
                assert before["param_groups"] == after["param_groups"]
                for parameter_id, state in before["state"].items():
                    for key, value in state.items():
                        torch.testing.assert_close(value, after["state"][parameter_id][key], atol=0, rtol=0)
        sink = ScalarSink(output=resumed.output)
        try:
            result = resumed.run(sink)
        finally:
            sink.close()
        assert result.name == "00000008" and resumed.updates == 4
        assert resumed.output.parent == resume_cfg.output
        assert (resumed.output / "checkpoints/00000006/config.json").is_file()
        rows = [json.loads(line) for line in (resumed.output / "metrics.jsonl").read_text().splitlines()]
        assert [row["iteration"] for row in rows] == [5, 6, 7, 8]
        assert "wm_loss" not in rows[0] and "wm_loss" in rows[-1]
        assert (resumed.output / "training-curves.png").is_file()
        assert "截至迭代 8" in (resumed.output / "training-curves.svg").read_text()
        inherited = parse_command(["train", "--resume", str(saved)])
        assert isinstance(inherited, TrainCmd) and inherited.output == cfg.output


def test_training_artifacts_and_refresh_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """真实 CPU 训练覆盖时间戳、逐轮记录、周期刷新和不足周期的最终保存。"""
    cfg = replace(_small_config(tmp_path), max_iterations=5)
    refreshes: list[int] = []
    render = training.plot_training_curves

    def capture(rows: list[dict[str, float]], output: Path, labels: dict[str, str]) -> None:
        iteration = int(rows[-1]["iteration"])
        assert (output / "checkpoints" / f"{iteration:08d}" / "config.json").is_file()
        refreshes.append(iteration)
        render(rows, output, labels)

    monkeypatch.setattr(training, "plot_training_curves", capture)
    result = training.train(cfg)
    assert result.parent.name == "checkpoints"
    output = result.parent.parent
    assert output.parent == cfg.output and re.fullmatch(r"\d{8}_\d{6}(?:_\d+)?", output.name)
    assert result.name == "00000005"
    assert refreshes == [2, 4, 5]
    assert {path.name for path in output.iterdir()} == {
        "config.json",
        "run.json",
        "checkpoints",
        "evaluations",
        "exports",
        "metrics.jsonl",
        "training-curves.png",
        "training-curves.svg",
    }
    assert {path.name for path in (output / "checkpoints").iterdir()} == {
        "00000002",
        "00000004",
        "00000005",
        "index.json",
    }
    assert json.loads((output / "checkpoints/index.json").read_text())["latest"] == result.name
    assert json.loads((output / "run.json").read_text())["resume"] is None
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[2]["wm_loss"] > 0 and "wm_loss" not in rows[0]
    assert (output / "training-curves.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "截至迭代 5" in (output / "training-curves.svg").read_text()
    assert load_config(result).output == cfg.output
    console = capsys.readouterr().out
    assert re.findall(r"^迭代 (\d+) \|", console, re.MULTILINE) == ["2", "4"]
    assert f"训练输出目录：{output}" in console


def test_default_config_and_invalid_values(tmp_path: Path) -> None:
    cfg = TrainConfig()
    assert cfg.num_envs == 4096 and cfg.max_iterations == 100_000
    assert cfg.log_interval == 100 and cfg.save_interval == 1000
    assert cfg.buffer_capacity == 1000 and cfg.replay_ratio == 16
    assert (cfg.wm.batch, cfg.wm.horizon, cfg.wm.num_batches) == (2048, 8, 10)
    assert (cfg.policy.batch, cfg.policy.horizon) == (2048, 32)
    assert cfg.dynamics.hidden == (2048, 2048, 2048)
    assert (cfg.policy.critic_epochs, cfg.policy.critic_minibatches) == (5, 4)
    assert cfg.compile and cfg.policy.checkpoint
    with pytest.raises(ValueError, match="motion-file"):
        cfg.validate()
    with pytest.raises(ValueError, match="horizon"):
        replace(_small_config(tmp_path), buffer_capacity=2).validate()
    with pytest.raises(ValueError, match="gamma"):
        replace(_small_config(tmp_path), policy=PolicyTrainConfig(gamma=1)).validate()

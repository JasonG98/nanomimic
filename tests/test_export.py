"""纯 Actor 的观测布局、ONNX 数值与部署元数据验收。

用递增值独立校验观测布局，并直接对照 PyTorch Actor。
导出走正常检查点入口，配置指向一个不存在的动作文件。
"""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from nanomimic.cli import ExportCmd, parse_command
from nanomimic.common.specs import SimConfig
from nanomimic.models.policy import Actor, ActorConfig
from nanomimic.runtime.export import export, term_major_to_history
from nanomimic.runtime.inference import OnnxActor, deployment_obs
from nanomimic.runtime.sim import Sim
from nanomimic.training.config import TrainConfig, config_dict


@pytest.mark.parametrize("length", [1, 3, 5])
def test_term_major_layout_uses_oldest_frame_first(length: int) -> None:
    history = torch.arange(2 * length * 93).reshape(2, length, 93).float()
    # 独立按项、时间、通道三层顺序拼接，避免往返转换掩盖布局错误。
    columns = [(0, 3), (3, 6), (6, 35), (35, 64), (64, 93)]
    packed = torch.stack(
        [torch.cat([history[row, step, lo:hi] for lo, hi in columns for step in range(length)]) for row in range(2)]
    )
    torch.testing.assert_close(term_major_to_history(packed, 29, length), history, atol=0, rtol=0)
    goal = torch.arange(1280).reshape(2, 640).float()
    torch.testing.assert_close(deployment_obs(history, goal), torch.cat((packed, goal), dim=1), atol=0, rtol=0)


def test_onnx_export_without_motion_file(tmp_path: Path) -> None:
    torch.manual_seed(82)
    actor_cfg = ActorConfig(history_length=3, hidden=(16, 12, 8, 8))
    actor = Actor(29, actor_cfg).eval()
    actor.obs_norm.update(torch.randn(12, 93) + 0.7)
    actor.goal_norm.update(torch.randn(12, 640) - 0.4)
    checkpoint = tmp_path / "run/checkpoints/00000042"
    checkpoint.mkdir(parents=True)
    cfg = TrainConfig(motion_file=tmp_path / "missing.npz", actor=actor_cfg)
    (checkpoint / "config.json").write_text(json.dumps(config_dict(cfg)))
    torch.save({"schema_version": 1, "iteration": 42, "model": actor.state_dict()}, checkpoint / "actor.pt")

    export(checkpoint, verify=True)
    out = tmp_path / "run/exports/actor-00000042.onnx"
    assert list(out.parent.iterdir()) == [out]
    model = onnx.load(out)
    assert [value.name for value in model.graph.input] == ["obs"]
    assert [value.name for value in model.graph.output] == ["actions"]
    assert any(op.domain in ("", "ai.onnx") and op.version == 18 for op in model.opset_import)
    assert all(value.data_location == onnx.TensorProto.DEFAULT for value in model.graph.initializer)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert [(item.shape, item.type) for item in session.get_inputs()] == [([1, 919], "tensor(float)")]
    assert [(item.shape, item.type) for item in session.get_outputs()] == [([1, 29], "tensor(float)")]

    onnx_actor = OnnxActor(out)
    assert onnx_actor.history_length == 3
    assert onnx_actor.iteration == 42
    # 两行采用不同目标，同一个模型应正确处理两份外部输入。
    history = torch.randn(2, 3, 93)
    goal = torch.randn(2, 640)
    with torch.no_grad():
        error = float((onnx_actor(history, goal) - actor(history, goal)).abs().max())
    assert error < 1e-4

    meta = session.get_modelmeta().custom_metadata_map
    assert meta["run_path"] == str(checkpoint)
    motion_file = Path(__file__).parent / "fixtures/motion.npz"
    with Sim(SimConfig(motion_file, num_envs=1, device="cpu")) as sim:
        assert onnx_actor.joint_names == sim.robot.joint_names
        assert json.loads(meta["motion_body_names"]) == list(sim.robot.tree_body_names)
        assert meta["anchor_body_name"] == sim.robot.anchor_body_name
        assert float(meta["control_dt"]) == sim.robot.dt
        for key in ("default_joint_pos", "action_scale", "action_offset"):
            np.testing.assert_array_equal(json.loads(meta[key]), getattr(sim.robot, key).numpy())


def test_export_cli_options() -> None:
    checkpoint = "runs/g1/20261118_120000/checkpoints/00000100"
    command = parse_command(["export", "--checkpoint", checkpoint, "--out", "outputs/g1.onnx", "--verify"])
    assert isinstance(command, ExportCmd) and command.verify
    assert command.checkpoint == Path(checkpoint) and command.out == Path("outputs/g1.onnx")
    default = parse_command(["export", "--checkpoint", checkpoint])
    assert isinstance(default, ExportCmd) and default.out is None

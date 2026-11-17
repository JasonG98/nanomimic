"""T8 的布局、动作库、ONNX 数值与部署元数据验收。

布局用递增哨兵独立构造，数值直接对照 Actor，不以包装器自身作为真值。
动作库使用既有只读夹具；帧数边界不改变夹具或降低误差阈值。
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from nanomimic.cli import ExportCmd, parse_command
from nanomimic.common.specs import Motion, SimConfig
from nanomimic.models.policy import Actor, ActorConfig
from nanomimic.runtime.export import MotionPolicy, deployment_obs, term_major_to_history, verify_onnx, write_onnx
from nanomimic.runtime.sim import Sim


@pytest.mark.parametrize("length", [1, 3, 5])
def test_term_major_layout_uses_oldest_frame_first(length: int) -> None:
    history = torch.arange(2 * length * 93).reshape(2, length, 93).float()
    # 独立按 term、时间、通道三层顺序拼接，避免往返测试掩盖同一布局错误。
    columns = [(0, 3), (3, 6), (6, 35), (35, 64), (64, 93)]
    packed = torch.stack(
        [torch.cat([history[row, step, lo:hi] for lo, hi in columns for step in range(length)]) for row in range(2)]
    )
    torch.testing.assert_close(term_major_to_history(packed, 29, length), history, atol=0, rtol=0)
    goal = torch.arange(1280).reshape(2, 640).float()
    torch.testing.assert_close(deployment_obs(history, goal), torch.cat((packed, goal), dim=-1), atol=0, rtol=0)


def test_onnx_contract_constants_clamping_and_cpu_error(tmp_path: Path, motion: Motion) -> None:
    torch.manual_seed(82)
    actor = Actor(29, ActorConfig(history_length=3, hidden=(16, 12, 8, 8)))
    actor.obs_norm.update(torch.randn(12, 93) + 0.7)
    actor.goal_norm.update(torch.randn(12, 640) - 0.4)
    before = {name: value.clone() for name, value in actor.state_dict().items()}
    policy = MotionPolicy(actor, motion)
    out = tmp_path / "policy.onnx"
    write_onnx(policy, out)
    error = verify_onnx(policy, out, motion.clip_ends)
    assert error < 1e-4
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert [(item.name, item.shape, item.type) for item in session.get_inputs()] == [
        ("obs", [1, 919], "tensor(float)"),
        ("time_step", [1, 1], "tensor(float)"),
    ]
    assert [item.name for item in session.get_outputs()] == policy.output_names()
    model = onnx.load(out)
    # 动作库必须以常量内嵌在单个文件里，而不是外部数据。
    assert all(value.data_location == onnx.TensorProto.DEFAULT for value in model.graph.initializer)
    initializers = {value.name: onnx.numpy_helper.to_array(value) for value in model.graph.initializer}
    for name in policy.output_names()[1:]:
        # 导出器会重命名常量；按图连线找到每个输出所用的首个输入，再核对内容。
        node = next(node for node in model.graph.node if name in node.output)
        assert node.input[0] in initializers, name
        np.testing.assert_array_equal(initializers[node.input[0]], getattr(motion, name).numpy())
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_export_frame_limit(motion: Motion) -> None:
    actor = Actor(29, ActorConfig(hidden=(8,)))
    arrays = {
        name: getattr(motion, name)[:1].expand(15000, *getattr(motion, name).shape[1:])
        for name in MotionPolicy.output_names()[1:]
    }
    at_limit = replace(motion, **arrays, clip_ends=torch.tensor([14999]))
    assert MotionPolicy(actor, at_limit).frames == 15000
    for frames in (0, 15001):
        invalid = replace(motion, joint_pos=motion.joint_pos[:1].expand(frames, -1))
        with pytest.raises(ValueError, match="15000"):
            MotionPolicy(actor, invalid)


def test_metadata_matches_input_and_robot(tmp_path: Path) -> None:
    motion_file = Path(__file__).parent / "fixtures/motion.npz"
    with Sim(SimConfig(motion_file, num_envs=1, device="cpu")) as sim:
        policy = MotionPolicy(Actor(29, ActorConfig(history_length=3, hidden=(8,))), sim.motion)
        out = tmp_path / "metadata.onnx"
        write_onnx(policy, out)
        sim.attach_export_metadata(out, 3, "CPU 验收")
        meta = ort.InferenceSession(str(out)).get_modelmeta().custom_metadata_map
        assert meta["joint_names"].split(",") == list(sim.robot.joint_names)
        assert meta["body_names"].split(",") == list(sim.robot.body_names)
        assert meta["anchor_body_name"] == sim.robot.anchor_body_name
        assert meta["observation_names"].split(",") == [
            "projected_gravity",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
            "command",
            "motion_anchor_ori_b",
        ]
        assert meta["observation_history_lengths"].split(",") == ["3.000"] * 5 + ["1.000"] * 2
        for key in ("observation_terms_scale", "observation_terms_clip", "observation_terms_history_length"):
            assert len(meta[key].split(",")) == 7
        assert meta["run_path"] == "CPU 验收"
        assert len(meta["action_scale"].split(",")) == 29


def test_export_cli_options() -> None:
    command = parse_command(["export", "--checkpoint", "runs/100", "--out", "outputs/g1.onnx", "--verify"])
    assert isinstance(command, ExportCmd) and command.verify
    assert command.checkpoint == Path("runs/100") and command.out == Path("outputs/g1.onnx")
    default = parse_command(["export", "--checkpoint", "runs/100"])
    assert isinstance(default, ExportCmd) and default.out is None

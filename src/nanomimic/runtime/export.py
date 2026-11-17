"""Actor 与本地动作库的固定批量 ONNX 部署图。

输入历史为 term 主序，图内转换成 Actor 使用的帧主序；目标不堆叠历史。
动作库作为常量嵌入，时间钳在有效范围；CPU 自检不代替最终 sim2sim 验收。
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from nanomimic.common.specs import Motion, SimConfig
from nanomimic.models.policy import Actor
from nanomimic.runtime.paths import resolve_checkpoint, run_directory
from nanomimic.runtime.sim import Sim
from nanomimic.training.checkpoints import load_weights
from nanomimic.training.config import load_config


def term_major_to_history(obs: Tensor, joints: int, length: int) -> Tensor:
    """[B, H×D] 的逐 term 历史转成 [B, H, D]；每个 term 内最旧在前。"""
    terms, offset = [], 0
    for width in (3, 3, joints, joints, joints):
        terms.append(obs[:, offset : offset + length * width].reshape(-1, length, width))
        offset += length * width
    return torch.cat(terms, dim=-1)


def deployment_obs(history: Tensor, goal: Tensor) -> Tensor:
    """把帧主序历史和 actor_goal 打包成部署输入 [B, H×D+G]。"""
    joints = (history.shape[-1] - 6) // 3
    terms = history.split((3, 3, joints, joints, joints), dim=-1)
    return torch.cat((*[term.flatten(1) for term in terms], goal), dim=-1)


class MotionPolicy(nn.Module):
    """无内部历史状态；调用方维护 obs 历史与目标，time_step 只查询动作库。"""

    joint_pos: Tensor
    joint_vel: Tensor
    body_pos_w: Tensor
    body_quat_w: Tensor
    body_lin_vel_w: Tensor
    body_ang_vel_w: Tensor

    def __init__(self, actor: Actor, motion: Motion) -> None:
        super().__init__()
        if not 1 <= motion.num_frames <= 15000:
            raise ValueError("ONNX 导出要求动作库为 1 到 15000 帧")
        self.actor = deepcopy(actor).cpu().eval()
        self.joints = motion.joint_pos.shape[1]
        self.frames = motion.num_frames
        self.history_size = actor.history_length * (6 + 3 * self.joints)
        self.input_size = self.history_size + 10 * (2 * self.joints + 6)
        for name in self.output_names()[1:]:
            self.register_buffer(name, getattr(motion, name).detach().cpu().clone())

    @staticmethod
    def output_names() -> list[str]:
        # mjlab 部署契约的动作输出名为 actions。
        return ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]

    def forward(self, obs: Tensor, time_step: Tensor) -> tuple[Tensor, ...]:
        history = term_major_to_history(obs[:, : self.history_size], self.joints, self.actor.history_length)
        time = time_step.long().squeeze(-1).clamp(0, self.frames - 1)
        return (
            self.actor(history, obs[:, self.history_size :]),
            self.joint_pos[time],
            self.joint_vel[time],
            self.body_pos_w[time],
            self.body_quat_w[time],
            self.body_lin_vel_w[time],
            self.body_ang_vel_w[time],
        )


def write_onnx(policy: MotionPolicy, out: Path) -> None:
    """使用参考部署契约的 opset 18、fp32 与固定批量 1，产物只有一个 ONNX 文件。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        policy.eval(),
        (torch.zeros(1, policy.input_size), torch.zeros(1, 1)),
        str(out),
        opset_version=18,
        input_names=["obs", "time_step"],
        output_names=policy.output_names(),
        dynamo=True,
    )
    onnx.checker.check_model(str(out))


@torch.no_grad()
def verify_onnx(policy: MotionPolicy, out: Path, clip_ends: Tensor) -> float:
    """直接对照 Actor 与六个原始动作张量，覆盖随机历史、片段边界和越界时间。"""
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(0)
    times = [-1, 0, policy.frames - 1, policy.frames, policy.frames + 100]
    times += [time + delta for time in clip_ends.cpu().tolist() for delta in (-1, 0, 1)]
    times += torch.randint(policy.frames, (32,), generator=generator).tolist()
    error = 0.0
    for time in times:
        history = torch.randn(1, policy.actor.history_length, 6 + 3 * policy.joints, generator=generator)
        goal = torch.randn(1, 10 * (2 * policy.joints + 6), generator=generator)
        obs = deployment_obs(history, goal)
        outputs = session.run(None, {"obs": obs.numpy(), "time_step": np.array([[time]], dtype=np.float32)})
        expected = policy.actor(history, goal).numpy()
        difference = float(np.max(np.abs(outputs[0] - expected)))
        if not np.isfinite(difference) or difference >= 1e-4:
            raise ValueError(f"ONNX 动作自检失败：最大绝对误差 {difference:.6g}，要求小于 1e-4")
        error = max(error, difference)
        index = max(0, min(time, policy.frames - 1))
        for name, value in zip(policy.output_names()[1:], outputs[1:], strict=True):
            if not isinstance(value, np.ndarray) or not np.array_equal(
                value, getattr(policy, name)[index : index + 1].numpy()
            ):
                raise ValueError(f"ONNX 动作库自检失败：{name}，time_step={time}")
    return error


def export(checkpoint: Path, out: Path | None = None, verify: bool = False) -> None:
    """读取检查点配置与原动作文件，在 CPU 导出；不恢复优化器或启动训练。"""
    checkpoint = resolve_checkpoint(checkpoint)
    cfg = load_config(checkpoint)
    if cfg.motion_file is None:
        raise ValueError("检查点缺少 motion_file")
    with Sim(SimConfig(cfg.motion_file, num_envs=1, device="cpu", seed=cfg.seed)) as sim:
        actor = Actor(len(sim.robot.joint_names), cfg.actor)
        iteration = load_weights(checkpoint, "actor", actor, sim.device)
        out = out or run_directory(checkpoint) / "exports" / f"actor-{iteration:08d}.onnx"
        policy = MotionPolicy(actor, sim.motion)
        write_onnx(policy, out)
        sim.attach_export_metadata(out, actor.history_length, str(checkpoint.resolve()))
        if verify:
            error = verify_onnx(policy, out, sim.motion.clip_ends)
            print(f"CPU 数值自检通过：ONNX 与 PyTorch 动作最大绝对误差 {error:.6g} < 1e-4。")
            print("完整动作闭环且不摔倒的 sim2sim 验收，待 T7 得到合格策略后完成。")
    print(f"已导出迭代 {iteration} 的 Actor 与 {policy.frames} 帧动作库：{out}")

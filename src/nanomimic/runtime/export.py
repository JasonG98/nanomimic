"""从检查点导出固定批量的纯 Actor ONNX。

输入为按观测项排列的历史及外部构造的参考目标，图内包含归一化。
导出读取检查点与 G1 部署参数，支持 CPU 自检。
"""

import math
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import torch
from torch import Tensor, nn

from nanomimic.models.policy import Actor
from nanomimic.runtime.inference import OnnxActor
from nanomimic.runtime.paths import run_directory
from nanomimic.runtime.sim import deployment_metadata
from nanomimic.training.checkpoints import load_weights
from nanomimic.training.config import load_config


def term_major_to_history(obs: Tensor, joints: int, length: int) -> Tensor:
    """[B, H×D] 的逐项历史转成 [B, H, D]，每项内最旧帧在前。"""
    terms, offset = [], 0
    for width in (3, 3, joints, joints, joints):
        terms.append(obs[:, offset : offset + length * width].reshape(-1, length, width))
        offset += length * width
    return torch.cat(terms, dim=-1)


class DeploymentActor(nn.Module):
    """把部署观测解包为训练时的历史和目标，返回原始动作。"""

    def __init__(self, actor: Actor, joints: int) -> None:
        super().__init__()
        self.actor = deepcopy(actor).cpu().eval()
        self.joints = joints
        self.history_size = actor.history_length * (6 + 3 * joints)
        self.input_size = self.history_size + 10 * (2 * joints + 6)

    def forward(self, obs: Tensor) -> Tensor:
        history = term_major_to_history(obs[:, : self.history_size], self.joints, self.actor.history_length)
        return self.actor(history, obs[:, self.history_size :])


def _write_onnx(policy: DeploymentActor, out: Path, metadata: dict[str, str]) -> None:
    torch.onnx.export(
        policy.eval(),
        (torch.zeros(1, policy.input_size, dtype=torch.float32),),
        str(out),
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamo=True,
        external_data=False,
    )
    model = onnx.load(out)
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, out)


@torch.no_grad()
def verify_onnx(policy: DeploymentActor, out: Path) -> float:
    """对同样的历史与参考目标比较 ONNX 和 PyTorch Actor。"""
    actor = OnnxActor(out)
    generator = torch.Generator().manual_seed(0)
    error = 0.0
    for _ in range(32):
        history = torch.randn(1, policy.actor.history_length, 6 + 3 * policy.joints, generator=generator)
        goal = torch.randn(1, 10 * (2 * policy.joints + 6), generator=generator)
        outputs = actor(history, goal)
        expected = policy.actor(history, goal)
        difference = float((outputs - expected).abs().max())
        if not math.isfinite(difference) or difference >= 1e-4:
            raise ValueError(f"ONNX 动作自检失败：最大绝对误差 {difference:.6g}，要求小于 1e-4")
        error = max(error, difference)
    return error


def export(checkpoint: Path, out: Path | None = None, verify: bool = False) -> None:
    """从检查点配置与 Actor 权重生成单文件 ONNX。"""
    cfg = load_config(checkpoint)
    joints, metadata = deployment_metadata()
    actor = Actor(joints, cfg.actor)
    iteration = load_weights(checkpoint, "actor", actor, torch.device("cpu"))
    metadata.update(
        history_length=str(actor.history_length), iteration=str(iteration), run_path=str(checkpoint.resolve())
    )
    out = out or run_directory(checkpoint) / "exports" / f"actor-{iteration:08d}.onnx"
    policy = DeploymentActor(actor, joints)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".onnx-", dir=out.parent) as directory:
        staged = Path(directory) / out.name
        _write_onnx(policy, staged, metadata)
        if verify:
            error = verify_onnx(policy, staged)
            print(f"CPU 数值自检通过：ONNX 与 PyTorch 动作最大绝对误差 {error:.6g} < 1e-4。")
        staged.replace(out)
    print(f"已导出迭代 {iteration} 的 Actor：{out}")

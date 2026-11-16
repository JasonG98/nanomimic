"""四文件检查点的完整写入与恢复。

每个目录保存网络、归一化、优化器与排程进度，并附 schema_version 配置。
先写临时目录再重命名；恢复不保存物理引擎或回放缓冲，重新采集预热。
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from nanomimic.optim import Optimizers
from nanomimic.training_config import TrainConfig, config_dict


def save_checkpoint(
    directory: Path, cfg: TrainConfig, models: dict[str, tuple[nn.Module, Optimizers]], progress: dict[str, int | float]
) -> None:
    if directory.exists():
        raise FileExistsError(f"检查点目录已存在，拒绝覆盖：{directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".checkpoint-", dir=directory.parent) as temporary:
        stage = Path(temporary)
        for name, (model, optim) in models.items():
            payload = {
                "schema_version": 1,
                "iteration": progress["iteration"],
                "model": model.state_dict(),
                "optimizers": optim.state_dict(),
            }
            if name == "world_model":
                payload.update(progress=progress, rng_cpu=torch.get_rng_state())
                if next(model.parameters()).is_cuda:
                    payload["rng_cuda"] = torch.cuda.get_rng_state_all()
            torch.save(payload, stage / f"{name}.pt")
        (stage / "config.json").write_text(
            json.dumps(config_dict(cfg), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(directory)


def load_checkpoint(
    directory: Path, models: dict[str, tuple[nn.Module, Optimizers]], device: torch.device
) -> dict[str, int | float]:
    payloads = {name: torch.load(directory / f"{name}.pt", map_location=device, weights_only=True) for name in models}
    versions = {value.get("schema_version") for value in payloads.values()}
    iterations = {value.get("iteration") for value in payloads.values()}
    if versions != {1} or len(iterations) != 1:
        raise ValueError("检查点文件的版本或迭代号不一致")
    for name, (model, optim) in models.items():
        model.load_state_dict(payloads[name]["model"])
        optim.load_state_dict(payloads[name]["optimizers"])
    saved = payloads["world_model"]
    torch.set_rng_state(saved["rng_cpu"].cpu())
    if device.type == "cuda" and "rng_cuda" in saved:
        for index, state in enumerate(saved["rng_cuda"]):
            if index < torch.cuda.device_count():
                torch.cuda.set_rng_state(state.cpu(), index)
    return saved["progress"]

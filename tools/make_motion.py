"""把 LAFAN1 重定向动作转成 nanomimic 的 NPZ 动作文件。

做三件事：
1. 按片段名从 HuggingFace 下载 CSV，本地已有就跳过；
2. 用 mjlab 的 csv_to_npz 把 30 Hz 动作重采样到 50 Hz，并让 G1 模型做正运动学，
   算出每个 body 的世界系位置、四元数与速度；
3. 按 nanomimic 的字段名存成 NPZ。

用法：
    uv run python tools/make_motion.py run1_subject5
    uv run python tools/make_motion.py run1_subject5 --out data/lafan_run1_subject5.npz
"""

import sys
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro

# 数据源：40 段 LAFAN1 动作重定向到 Unitree G1，每段一个 CSV。
DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
CSV_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/g1"
# 仓库根下的 data/，CSV 与默认输出都放这里（体积大，已在 .gitignore 忽略）。
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Config:
    # 片段名（如 run1_subject5），或本地 CSV 路径。
    motion: Annotated[str, tyro.conf.Positional]
    # 输出 NPZ；默认 data/lafan_<片段名>.npz。
    out: Path | None = None


def download_csv(motion: str) -> Path:
    """片段名 → data/ 下的 CSV；本地已有就直接复用。"""
    csv_path = DATA_DIR / f"{Path(motion).stem}.csv"
    if not csv_path.is_file():
        url = f"{CSV_URL}/{csv_path.name}"
        print(f"下载 {url}")
        urllib.request.urlretrieve(url, csv_path)
    return csv_path


def _noop(*args: object, **kwargs: object) -> None:
    """什么也不做，用来替换 W&B 的上传调用。"""


class _SilentRun:
    """冒充 wandb.init() 的返回值，所有方法都是空操作。"""

    log = log_artifact = link_artifact = finish = staticmethod(_noop)


def silence_wandb() -> None:
    """mjlab 的 csv_to_npz 结尾会强制上传 W&B，这里塞一个空模块把它挡掉。"""
    module = types.ModuleType("wandb")
    module.__dict__.update({"init": lambda **kwargs: _SilentRun(), "finish": _noop})
    sys.modules["wandb"] = module


def convert(csv_path: Path, out: Path) -> None:
    """调 mjlab 的 csv_to_npz 做重采样与正运动学，再另存到目标路径。"""
    silence_wandb()
    from mjlab.scripts import csv_to_npz

    # 这个脚本把结果硬编码写到 /tmp/motion.npz，只能写完再读回来。
    csv_to_npz.main(input_file=str(csv_path), output_name=out.stem, device="cpu")
    data = dict(np.load("/tmp/motion.npz"))

    # 单个 CSV 就是一段完整动作，片段右开端点即总帧数。
    data["motion_ends"] = np.array([data["joint_pos"].shape[0]], dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)
    print(f"写出 {out}：{data['joint_pos'].shape[0]} 帧 @ {data['fps'].item():g} Hz")


def main(cfg: Config) -> None:
    csv_path = download_csv(cfg.motion)
    convert(csv_path, cfg.out or DATA_DIR / f"lafan_{csv_path.stem}.npz")


if __name__ == "__main__":
    main(tyro.cli(Config))

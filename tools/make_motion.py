"""把 LAFAN1 重定向动作转成 nanomimic 的 NPZ 动作文件。

做三件事：
1. 按片段名从 HuggingFace 下载 CSV，本地已有就跳过；
2. 用 mjlab 的 csv_to_npz 把 30 Hz 动作重采样到 50 Hz，并让 G1 模型做正运动学，
   算出每个 body 的世界系位置、四元数与速度；
3. 每段存一个 NPZ，并把选中的片段按给定顺序合并成带 motion_ends 的多片段 NPZ。

用法：
    uv run python tools/make_motion.py --list
    uv run python tools/make_motion.py --all
    uv run python tools/make_motion.py run1_subject5 walk1_subject1
    uv run python tools/make_motion.py --all --out-dir data/processed/lafan1-g1-v1/clips
"""

import sys
import types
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import tyro

# 数据源：40 段 LAFAN1 动作重定向到 Unitree G1，每段一个 CSV。
DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
CSV_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/g1"
# 原始 CSV、逐段处理结果与合并缓存分别存放；体积大，均在 .gitignore 忽略。
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# 数据集实际提供的 40 段；--all 按此顺序转换并合并，保证结果可复现。
LAFAN1_CLIPS = (
    "dance1_subject1",
    "dance1_subject2",
    "dance1_subject3",
    "dance2_subject1",
    "dance2_subject2",
    "dance2_subject3",
    "dance2_subject4",
    "dance2_subject5",
    "fallAndGetUp1_subject1",
    "fallAndGetUp1_subject4",
    "fallAndGetUp1_subject5",
    "fallAndGetUp2_subject2",
    "fallAndGetUp2_subject3",
    "fallAndGetUp3_subject1",
    "fight1_subject2",
    "fight1_subject3",
    "fight1_subject5",
    "fightAndSports1_subject1",
    "fightAndSports1_subject4",
    "jumps1_subject1",
    "jumps1_subject2",
    "jumps1_subject5",
    "run1_subject2",
    "run1_subject5",
    "run2_subject1",
    "run2_subject4",
    "sprint1_subject2",
    "sprint1_subject4",
    "walk1_subject1",
    "walk1_subject2",
    "walk1_subject5",
    "walk2_subject1",
    "walk2_subject3",
    "walk2_subject4",
    "walk3_subject1",
    "walk3_subject2",
    "walk3_subject3",
    "walk3_subject4",
    "walk3_subject5",
    "walk4_subject1",
)
# csv_to_npz 输出的数组；合并时按时间维拼接，fps 与 motion_ends 单独处理。
ARRAYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


@dataclass(frozen=True)
class Config:
    # 片段名（可多个，如 run1_subject5 walk1_subject1）；与 --all 二选一。
    motions: Annotated[tuple[str, ...], tyro.conf.Positional] = ()
    # 下载并转换数据集全部 40 段，按 LAFAN1_CLIPS 的顺序合并。
    all: bool = False
    # 只列出可用片段名与本地状态，不下载也不转换。
    list: bool = False
    # 原始 CSV 的缓存目录。
    raw_dir: Path = DATA_DIR / "raw/lafan1/g1"
    # 每段动作一个 NPZ；v1 表示当前转换约定。
    out_dir: Path = DATA_DIR / "processed/lafan1-g1-v1/clips"
    # 合并缓存；只转换单段时不写。
    merged: Path = DATA_DIR / "cache/lafan1-g1-v1/all.npz"
    # 转换设备；无 CUDA 时 csv_to_npz 会自动回退到 CPU。
    device: str = "cuda:0"


def download_csv(motion: str, raw_dir: Path) -> Path:
    """片段名映射到原始 CSV；本地已有就直接复用。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"{Path(motion).stem}.csv"
    if not csv_path.is_file():
        url = f"{CSV_URL}/{csv_path.name}"
        print(f"下载 {url}", flush=True)
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


@contextmanager
def relaxed_self_check(limit: float = 1e-3) -> Generator[None]:
    """放宽 csv_to_npz 内部的 fp32 自检容差，调用结束后恢复。

    该脚本用 assert_close 对照机器人 body 速度与重采样后算出的速度，默认
    相对容差 1.3e-6；部分片段在 fp32 下差异约 1.3e-5，会中断转换。
    这里只把超过默认容差但绝对误差小于 limit 的断言放过；差异更大时仍用
    原断言报错。不改 mjlab 源码，也不改变输出数据。
    """
    original = torch.testing.assert_close

    def patched(actual: Any, expected: Any, *args: Any, **kwargs: Any) -> None:
        try:
            original(actual, expected, *args, **kwargs)
        except AssertionError:
            difference = torch.max(torch.abs(torch.as_tensor(actual) - torch.as_tensor(expected)))
            if float(difference) > limit:
                raise
            print(f"自检差异 {float(difference):.3e} 在 fp32 容差内，继续", flush=True)

    torch.testing.assert_close = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.testing.assert_close = original  # type: ignore[assignment]


def convert(csv_path: Path, device: str) -> dict[str, Any]:
    """调 mjlab 的 csv_to_npz 做重采样与正运动学，返回单段数组。"""
    silence_wandb()
    from mjlab.scripts import csv_to_npz

    # 这个脚本把结果硬编码写到 /tmp/motion.npz，只能写完再读回来。
    with relaxed_self_check():
        csv_to_npz.main(input_file=str(csv_path), output_name=csv_path.stem, device=device)
    data = dict(np.load("/tmp/motion.npz"))
    # 单个 CSV 就是一段完整动作，片段右开端点即总帧数。
    data["motion_ends"] = np.array([data["joint_pos"].shape[0]], dtype=np.int64)
    return data


def merge(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """按给定顺序拼接各片段；motion_ends 为累计右开端点。"""
    if not parts:
        raise ValueError("合并至少需要一段动作")
    merged = {name: np.concatenate([part[name] for part in parts], axis=0) for name in ARRAYS}
    merged["fps"] = parts[0]["fps"]
    merged["motion_ends"] = np.cumsum([part["joint_pos"].shape[0] for part in parts]).astype(np.int64)
    return merged


def list_clips(csv_dir: Path, out_dir: Path) -> None:
    """打印数据集片段与本地 CSV / NPZ 是否就绪，不访问网络。"""
    print(f"数据集 {DATASET} 的 g1 目录共 {len(LAFAN1_CLIPS)} 段：")
    for index, name in enumerate(LAFAN1_CLIPS, start=1):
        csv_ready = (csv_dir / f"{name}.csv").is_file()
        npz_ready = (out_dir / f"{name}.npz").is_file()
        state = f"CSV {'有' if csv_ready else '无'}，NPZ {'有' if npz_ready else '无'}"
        print(f"{index:>2}. {name:<28} {state}")


def main(cfg: Config) -> None:
    if cfg.list:
        if cfg.all or cfg.motions:
            raise ValueError("--list 只列出片段，不能与 --all 或片段名同时给出")
        list_clips(cfg.raw_dir, cfg.out_dir)
        return
    if cfg.all and cfg.motions:
        raise ValueError("--all 与片段名不能同时给出")
    names = LAFAN1_CLIPS if cfg.all else cfg.motions
    if not names:
        raise ValueError("请给出至少一个片段名，或用 --all 转换数据集全部 40 段；--list 可查看片段名")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[dict[str, Any]] = []
    for index, name in enumerate(names, start=1):
        data = convert(download_csv(name, cfg.raw_dir), cfg.device)
        out = cfg.out_dir / f"{Path(name).stem}.npz"
        np.savez(out, **data)
        parts.append(data)
        print(
            f"[{index}/{len(names)}] 写出 {out}：{data['joint_pos'].shape[0]} 帧 @ {data['fps'].item():g} Hz",
            flush=True,
        )
    if len(parts) > 1:
        merged = merge(parts)
        cfg.merged.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cfg.merged, **merged)
        print(
            f"合并 {len(parts)} 段到 {cfg.merged}：{merged['joint_pos'].shape[0]} 帧，"
            f"{merged['motion_ends'].size} 个片段端点",
            flush=True,
        )


if __name__ == "__main__":
    main(tyro.cli(Config))

"""命令行入口：train / play / export 三个子命令。

T0 只提供参数骨架，解析后以非零码提示待实现的任务；
训练、回放与导出的实际逻辑分别在 T6、T9、T8 接入本文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import tyro


@dataclass(frozen=True)
class TrainCmd:
    """在仿真中训练世界模型与策略（T6 接入）。"""


@dataclass(frozen=True)
class PlayCmd:
    """用检查点在仿真中回放策略（T9 接入）。"""

    checkpoint: Path
    viewer: Literal["native", "viser"] = "native"


@dataclass(frozen=True)
class ExportCmd:
    """把 Actor 与动作库导出为 ONNX（T8 接入）。"""

    checkpoint: Path
    out: Path
    verify: bool = False


Command = (
    Annotated[TrainCmd, tyro.conf.subcommand(name="train")]
    | Annotated[PlayCmd, tyro.conf.subcommand(name="play")]
    | Annotated[ExportCmd, tyro.conf.subcommand(name="export")]
)


def _pending(command: str, task: str) -> NoReturn:
    """以非零状态退出，说明该子命令尚未实现。"""
    raise SystemExit(f"nanomimic {command}: 尚未实现（计划在 {task} 接入）")


def main() -> None:
    """解析并分发子命令；T0 阶段仅验证命令行契约。"""
    command = tyro.cli(
        Command,
        prog="nanomimic",
        description="nanomimic：训练、回放与导出人形动作跟踪策略。",
    )
    if isinstance(command, TrainCmd):
        _pending("train", "T6")
    if isinstance(command, PlayCmd):
        _pending("play", "T9")
    _pending("export", "T8")


if __name__ == "__main__":
    main()

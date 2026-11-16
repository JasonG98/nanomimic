"""命令行入口：train / play / export 三个子命令。

训练支持本地动作与断点恢复，参数由 frozen dataclass 经 tyro 暴露。
回放与导出的实际逻辑分别留给 T9、T8 接入本文件。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import tyro

from nanomimic.training.config import TrainConfig, load_config


@dataclass(frozen=True)
class TrainCmd(TrainConfig):
    """在仿真中训练世界模型与策略，CPU 验收需显式缩小配置。"""


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


def parse_command(arguments: list[str]) -> TrainCmd | PlayCmd | ExportCmd:
    """先读取恢复配置作为 tyro 默认值，确保显式参数包括默认值都可覆盖。"""
    if arguments[:1] == ["train"]:
        probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        probe.add_argument("--resume", type=lambda value: None if value == "None" else Path(value))
        known, _ = probe.parse_known_args(arguments[1:])
        saved = load_config(known.resume) if known.resume is not None else TrainConfig()
        default = TrainCmd(**{item.name: getattr(saved, item.name) for item in fields(saved)})
        return tyro.cli(TrainCmd, default=default, args=arguments[1:], prog="nanomimic train")
    return tyro.cli(
        Command, args=arguments, prog="nanomimic", description="nanomimic：训练、回放与导出人形动作跟踪策略。"
    )


def main() -> None:
    """解析并分发子命令。"""
    try:
        command = parse_command(sys.argv[1:])
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"读取配置失败：{error}") from error
    if isinstance(command, TrainCmd):
        from nanomimic.training.train import train

        try:
            train(command)
        except (ValueError, FileNotFoundError, FileExistsError) as error:
            raise SystemExit(f"训练失败：{error}") from error
        return
    if isinstance(command, PlayCmd):
        _pending("play", "T9")
    _pending("export", "T8")


if __name__ == "__main__":
    main()

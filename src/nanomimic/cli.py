"""命令行入口：train / play / export 三个子命令。

训练支持本地动作与断点恢复，参数由 frozen dataclass 经 tyro 暴露。
回放与导出分别由 runtime/viewer.py 和 runtime/export.py 执行。
帮助页按终端显示列宽排版，避免中文宽字符越过边框。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Annotated, Literal
from unittest.mock import patch

import tyro

from nanomimic.training.config import TrainConfig, load_config


@dataclass(frozen=True)
class TrainCmd(TrainConfig):
    """在仿真中训练世界模型与策略，CPU 验收需显式缩小配置。"""


@dataclass(frozen=True)
class PlayCmd:
    """用检查点回放策略，同时叠加 env 0 的世界模型闭环虚影。"""

    # 含 config.json、actor.pt 与 world_model.pt 的检查点目录。
    checkpoint: Path
    # 原生窗口或 Viser 浏览器窗口。
    viewer: Literal["native", "viser"] = "native"
    # 默认有 CUDA 就用 cuda:0，否则用 CPU。
    device: str | None = None
    # 定期重新对齐的步数；默认取参考回放配置。
    num_steps_rollout: int = 1_000_000
    # 达到指定步数后退出；默认一直回放直到关闭窗口或 Ctrl+C。
    num_steps: int | None = None


@dataclass(frozen=True)
class ExportCmd:
    """把 Actor 与动作库导出为 ONNX，可执行 CPU 数值一致性自检。"""

    # 含 config.json 与 actor.pt 的检查点目录。
    checkpoint: Path
    # 单文件 ONNX 路径；省略时写入运行目录的 exports/actor-迭代号.onnx。
    out: Path | None = None
    # 对照 PyTorch Actor 与原始动作库执行 CPU 数值自检。
    verify: bool = False


Command = (
    Annotated[TrainCmd, tyro.conf.subcommand(name="train")]
    | Annotated[PlayCmd, tyro.conf.subcommand(name="play")]
    | Annotated[ExportCmd, tyro.conf.subcommand(name="export")]
)


@contextmanager
def _unicode_help() -> Generator[None]:
    """局部适配 tyro 1.0 的字符计数排版，解析结束或退出时恢复原渲染器。"""
    # tyro 暂无公开字宽回调；兼容逻辑只在本次解析期间生效。
    from tyro import _fmtlib

    original_render = _fmtlib._Text.render
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    padding = re.compile(r" +(?=(?:\x1b\[[0-9;]*m)*$)")

    def cell_width(text: str) -> int:
        return sum(
            0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in "WF" else 1
            for char in ansi.sub("", text)
        )

    def text_width(text: _fmtlib._Text) -> int:
        return cell_width(text.as_str_no_ansi())

    def render(text: _fmtlib._Text, width: int | None = None) -> list[str]:
        if width is None:
            return original_render(text)
        wrap_width = width
        while True:
            # 沿用 tyro 的分词与颜色，只收紧换行预算并按实际列宽补空格。
            lines = [padding.sub("", line) for line in original_render(text, wrap_width)]
            widths = [cell_width(line) for line in lines]
            overflow = max(widths, default=0) - width
            if overflow <= 0 or wrap_width <= 1:
                return [line + " " * max(0, width - cells) for line, cells in zip(lines, widths, strict=True)]
            wrap_width = max(1, wrap_width - (overflow + 1) // 2)

    with patch.object(_fmtlib._Text, "__len__", text_width), patch.object(_fmtlib._Text, "render", render):
        yield


@_unicode_help()
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
        # macOS 原生窗口必须使用 MuJoCo 提供的主线程启动器。
        if sys.platform == "darwin" and command.viewer == "native" and "MJPYTHON_BIN" not in os.environ:
            launcher = Path(sys.executable).with_name("mjpython")
            # uv 的 venv 使用解释器符号链接，mjpython 需要显式找到基座的 libpython。
            fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{Path(sys.base_prefix) / 'lib'}:{fallback}"
            os.execv(str(launcher), [str(launcher), "-m", "nanomimic.cli", *sys.argv[1:]])
        from nanomimic.runtime.viewer import play

        try:
            play(command.checkpoint, command.viewer, command.device, command.num_steps_rollout, command.num_steps)
        except (ValueError, FileNotFoundError) as error:
            raise SystemExit(f"回放失败：{error}") from error
        if sys.platform == "darwin" and command.viewer == "native" and command.num_steps is not None:
            # mjpython 在发生回合重置后可能停在 Cocoa 原生退出阶段；此时两侧资源已显式关闭。
            # 仅限成功完成的有限步自动回放，交互式窗口保留发布版生命周期。
            os._exit(0)
        return
    from nanomimic.runtime.export import export

    try:
        export(command.checkpoint, command.out, command.verify)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"导出失败：{error}") from error


if __name__ == "__main__":
    main()

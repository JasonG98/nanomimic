"""把旧数据与运行产物移动到分层目录，不重写数值文件。

默认只展示迁移计划，--apply 才执行；执行前检查全部目标，拒绝覆盖。
移动后核对文件身份与大小，并记录原、新路径；旧动作路径保留相对软链接。
历史运行名保留；混有新时间戳运行的根目录将旧平铺产物收进 legacy/。
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import tyro

from nanomimic.runtime.paths import write_json


@dataclass(frozen=True)
class Config:
    # 仓库根目录；仅迁移其中的 data/ 与 runs/。
    root: Path = Path(__file__).resolve().parents[1]
    # 执行迁移；默认只显示计划。
    apply: bool = False


@dataclass(frozen=True)
class Move:
    source: Path
    target: Path
    alias: bool = False


def plan_migration(root: Path) -> tuple[list[Move], list[Path]]:
    """只读生成迁移计划及待整理的运行目录，忽略已迁移的软链接。"""
    data, runs = root / "data", root / "runs"
    moves = [Move(path, data / "raw/lafan1/g1" / path.name) for path in sorted(data.glob("*.csv"))]
    for source, target in (
        ("lafan", "processed/lafan1-g1-v1/clips"),
        ("lafan1_all.npz", "cache/lafan1-g1-v1/all.npz"),
        ("lafan_run1_subject5.npz", "processed/lafan1-g1-v1/legacy/lafan_run1_subject5.npz"),
    ):
        path = data / source
        if path.exists() and not path.is_symlink():
            moves.append(Move(path, data / target, alias=True))
    run_roots: list[Path] = []
    for experiment in sorted(runs.iterdir()) if runs.is_dir() else []:
        if not experiment.is_dir() or experiment.is_symlink():
            continue
        children = list(experiment.iterdir())
        timestamps = [path for path in children if path.is_dir() and re.fullmatch(r"\d{8}_\d{6}(?:_\d+)?", path.name)]
        candidates = [experiment, *timestamps]
        if (experiment / "legacy").is_dir():
            candidates.append(experiment / "legacy")
        for source in candidates:
            checkpoints = [
                path
                for path in source.iterdir()
                if path.is_dir() and path.name.isdigit() and (path / "config.json").is_file()
            ]
            if not checkpoints and not (source / "checkpoints").is_dir() and not (source / "metrics.jsonl").is_file():
                continue
            target = source / "legacy" if source == experiment and timestamps and checkpoints else source
            run_roots.append(target)
            moves.extend(Move(path, target / "checkpoints" / path.name) for path in sorted(checkpoints))
            moves.extend(Move(path, target / "evaluations" / path.name) for path in sorted(source.glob("eval-*.json")))
            moves.extend(Move(path, target / "exports" / path.name) for path in sorted(source.glob("*.onnx")))
    return moves, list(dict.fromkeys(run_roots))


def _snapshot(path: Path) -> dict[str, tuple[int, int, int]]:
    paths = path.rglob("*") if path.is_dir() else [path]
    return {
        str(item.relative_to(path)) if path.is_dir() else ".": (stat.st_dev, stat.st_ino, stat.st_size)
        for item in paths
        if item.is_file()
        for stat in [item.stat()]
    }


def _preflight(moves: list[Move]) -> None:
    for move in moves:
        if move.target.exists() or move.target.is_symlink():
            raise FileExistsError(f"迁移目标已存在，拒绝覆盖：{move.target}")
        if not move.source.exists() or move.source.is_symlink():
            raise FileNotFoundError(f"迁移源已变化，请重新生成计划：{move.source}")
        for parent in move.target.parents:
            if parent.exists() and not parent.is_dir():
                raise NotADirectoryError(f"迁移目标的父路径不是目录：{parent}")


def apply_moves(moves: list[Move]) -> list[dict[str, Any]]:
    """预检所有目标后执行同一文件系统内的移动，核对文件身份并保留兼容链接。"""
    _preflight(moves)
    records = []
    for move in moves:
        before = _snapshot(move.source)
        move.target.parent.mkdir(parents=True, exist_ok=True)
        move.source.rename(move.target)
        if _snapshot(move.target) != before:
            raise RuntimeError(f"移动后文件身份或大小变化：{move.target}")
        if move.alias:
            move.source.symlink_to(
                os.path.relpath(move.target, move.source.parent), target_is_directory=move.target.is_dir()
            )
        records.append(
            {
                "source": str(move.source),
                "target": str(move.target),
                "alias": move.alias,
                "files": len(before),
                "bytes": sum(stat[2] for stat in before.values()),
            }
        )
    return records


def migrate(cfg: Config) -> Path | None:
    """迁移目录并保存路径记录；不加载模型，不转换或删除任何数值产物。"""
    root = cfg.root.resolve()
    moves, run_roots = plan_migration(root)
    if not cfg.apply:
        for move in moves:
            print(f"计划：{move.source.relative_to(root)} → {move.target.relative_to(root)}")
        print(f"共 {len(moves)} 项移动、{len(run_roots)} 个运行目录；加 --apply 执行。")
        return None
    _preflight(moves)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    report = root / "data/migrations" / f"{now:%Y%m%d_%H%M%S_%f}.json"
    plan = {
        "schema_version": 1,
        "migrated_at": now.isoformat(),
        "status": "planned",
        "moves": [{"source": str(move.source), "target": str(move.target), "alias": move.alias} for move in moves],
    }
    if moves:
        # 先保留完整路径映射，即使执行中断也能找到每个产物的旧、新位置。
        write_json(report, plan)
    records = apply_moves(moves)
    for run in run_roots:
        for name in ("checkpoints", "evaluations", "exports"):
            (run / name).mkdir(parents=True, exist_ok=True)
        checkpoints = [path for path in (run / "checkpoints").iterdir() if path.is_dir() and path.name.isdigit()]
        if checkpoints:
            latest = max(checkpoints, key=lambda path: int(path.name))
            if not (run / "config.json").exists():
                (run / "config.json").write_bytes((latest / "config.json").read_bytes())
            write_json(run / "checkpoints/index.json", {"schema_version": 1, "latest": latest.name})
        if not (run / "run.json").exists():
            write_json(run / "run.json", {"schema_version": 1, "migrated_at": now.isoformat()})
    if not records:
        print("目录已整理，无需移动文件。")
        return None
    write_json(report, {**plan, "status": "complete", "moves": records})
    print(f"已移动 {len(records)} 项，保留全部数值文件；迁移记录：{report}")
    return report


if __name__ == "__main__":
    migrate(tyro.cli(Config, description="整理 data 与 runs 的目录结构；默认只显示计划，--apply 执行。"))

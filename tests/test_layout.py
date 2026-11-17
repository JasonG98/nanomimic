"""目录迁移与旧检查点定位的兼容验收。

只使用临时目录，检查迁移前后文件内容、旧动作路径和重复执行行为。
目标冲突必须在任何移动前报错；测试不接触真实训练产物。
"""

import hashlib
import json
from pathlib import Path
from runpy import run_path

import pytest

from nanomimic.runtime.paths import resolve_checkpoint, run_directory

tool = run_path(str(Path(__file__).parents[1] / "tools/migrate_layout.py"))


def _old_layout(root: Path) -> tuple[Path, dict[Path, bytes]]:
    checkpoint = root / "runs/experiment/00001000"
    files = {
        root / "data/run1_subject5.csv": b"raw,values\n",
        root / "data/lafan/run1_subject5.npz": b"processed motion",
        root / "data/lafan1_all.npz": b"merged motion",
        root / "data/lafan_run1_subject5.npz": b"legacy motion",
        checkpoint / "config.json": b'{"schema_version": 1}\n',
        checkpoint / "actor.pt": b"actor and normalization",
        checkpoint / "world_model.pt": b"world model and optimizer",
        checkpoint / "critic.pt": b"critic and optimizer",
        root / "runs/experiment/eval-00001000-seed0.json": b'{"iteration": 1000}\n',
    }
    for path, value in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    return checkpoint, files


def test_migration_preserves_bytes_and_old_motion_paths(tmp_path: Path) -> None:
    checkpoint, files = _old_layout(tmp_path)
    old_hashes = {path: hashlib.sha256(value).hexdigest() for path, value in files.items()}
    report = tool["migrate"](tool["Config"](root=tmp_path, apply=True))
    records = json.loads(report.read_text())["moves"]
    for source, digest in old_hashes.items():
        move = next(row for row in records if source == Path(row["source"]) or Path(row["source"]) in source.parents)
        target = Path(move["target"])
        if source != Path(move["source"]):
            target = target / source.relative_to(move["source"])
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    assert (tmp_path / "data/lafan1_all.npz").read_bytes() == b"merged motion"
    assert (tmp_path / "data/lafan_run1_subject5.npz").read_bytes() == b"legacy motion"
    assert (tmp_path / "data/lafan/run1_subject5.npz").read_bytes() == b"processed motion"
    assert (tmp_path / "data/raw/lafan1/g1/run1_subject5.csv").read_bytes() == b"raw,values\n"
    migrated = checkpoint.parent / "checkpoints" / checkpoint.name
    assert resolve_checkpoint(checkpoint) == migrated
    assert resolve_checkpoint(migrated) == migrated
    assert run_directory(checkpoint) == checkpoint.parent
    assert (checkpoint.parent / "evaluations/eval-00001000-seed0.json").is_file()
    assert json.loads((checkpoint.parent / "checkpoints/index.json").read_text())["latest"] == checkpoint.name
    assert (checkpoint.parent / "config.json").read_bytes() == files[checkpoint / "config.json"]
    assert tool["plan_migration"](tmp_path)[0] == []
    assert tool["migrate"](tool["Config"](root=tmp_path, apply=True)) is None


def test_collision_and_dry_run_leave_sources_untouched(tmp_path: Path) -> None:
    _, files = _old_layout(tmp_path)
    assert tool["migrate"](tool["Config"](root=tmp_path)) is None
    assert all(path.read_bytes() == value for path, value in files.items())
    target = tmp_path / "data/cache/lafan1-g1-v1/all.npz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"another motion")
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        tool["migrate"](tool["Config"](root=tmp_path, apply=True))
    assert all(path.read_bytes() == value for path, value in files.items())
    assert target.read_bytes() == b"another motion"


def test_mixed_experiment_keeps_timestamp_run_separate(tmp_path: Path) -> None:
    checkpoint, _ = _old_layout(tmp_path)
    timestamp_run = checkpoint.parent / "20261118_120000"
    timestamp_run.mkdir()
    (timestamp_run / "metrics.jsonl").write_text('{"iteration": 1}\n')
    tool["migrate"](tool["Config"](root=tmp_path, apply=True))
    assert resolve_checkpoint(checkpoint) == checkpoint.parent / "legacy/checkpoints" / checkpoint.name
    assert run_directory(checkpoint) == checkpoint.parent / "legacy"
    assert (timestamp_run / "metrics.jsonl").read_text() == '{"iteration": 1}\n'
    assert all((timestamp_run / name).is_dir() for name in ("checkpoints", "evaluations", "exports"))
    assert tool["plan_migration"](tmp_path)[0] == []

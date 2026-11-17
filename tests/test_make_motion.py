"""动作转换工具的合并逻辑测试。

只测纯函数：多片段按给定顺序拼接、累计右开端点与总帧数一致。
不下载数据、不调用 mjlab 转换，也不修改受版本控制的夹具。
"""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

tool = run_path(str(Path(__file__).parents[1] / "tools/make_motion.py"))
merge = tool["merge"]
list_clips = tool["list_clips"]
ARRAYS = tool["ARRAYS"]
LAFAN1_CLIPS = tool["LAFAN1_CLIPS"]


def _part(frames: int, offset: float) -> dict[str, np.ndarray]:
    """构造一段与 csv_to_npz 输出同形状的假数据。"""
    joint = np.full((frames, 29), offset, dtype=np.float32)
    body = np.full((frames, 30, 3), offset, dtype=np.float32)
    quat = np.zeros((frames, 30, 4), dtype=np.float32)
    quat[..., 0] = 1
    return {
        "joint_pos": joint,
        "joint_vel": joint,
        "body_pos_w": body,
        "body_quat_w": quat,
        "body_lin_vel_w": body,
        "body_ang_vel_w": body,
        "fps": np.array([50.0]),
        "motion_ends": np.array([frames], dtype=np.int64),
    }


def test_merge_concatenates_in_order_and_accumulates_ends() -> None:
    merged = merge([_part(5, 1.0), _part(3, 2.0), _part(4, 3.0)])
    assert set(merged) == {*ARRAYS, "fps", "motion_ends"}
    assert merged["joint_pos"].shape == (12, 29)
    # 顺序必须保持输入顺序，端点累计且最后一个等于总帧数。
    assert merged["joint_pos"][[0, 5, 8], 0].tolist() == [1.0, 2.0, 3.0]
    assert merged["motion_ends"].tolist() == [5, 8, 12]
    assert merged["motion_ends"].dtype == np.int64
    assert float(merged["fps"][0]) == 50.0


def test_merge_rejects_empty() -> None:
    with pytest.raises(ValueError, match="至少需要一段"):
        merge([])


def test_list_clips_reports_local_state_without_network(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--list 只读本地文件：列出全部片段，并区分就绪与未就绪。"""
    assert len(LAFAN1_CLIPS) == 40
    csv_dir, out_dir = tmp_path / "csv", tmp_path / "npz"
    csv_dir.mkdir()
    list_clips(csv_dir, out_dir)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(LAFAN1_CLIPS) + 1
    assert lines[1].endswith(f"{LAFAN1_CLIPS[0]:<28} CSV 无，NPZ 无")
    assert lines[-1].endswith(f"{LAFAN1_CLIPS[-1]:<28} CSV 无，NPZ 无")
    # 放好第一段的 CSV 与 NPZ 后，只有该行显示就绪。
    out_dir.mkdir()
    (csv_dir / f"{LAFAN1_CLIPS[0]}.csv").write_text("")
    (out_dir / f"{LAFAN1_CLIPS[0]}.npz").write_bytes(b"")
    list_clips(csv_dir, out_dir)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].endswith("CSV 有，NPZ 有")
    assert lines[2].endswith("CSV 无，NPZ 无")


def test_conversion_uses_separate_raw_processed_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """不联网地运行转换入口，检查新目录与合并内容，保留原始 CSV。"""
    cfg = tool["Config"](
        motions=LAFAN1_CLIPS[:2],
        raw_dir=tmp_path / "raw/lafan1/g1",
        out_dir=tmp_path / "processed/lafan1-g1-v1/clips",
        merged=tmp_path / "cache/lafan1-g1-v1/all.npz",
    )
    cfg.raw_dir.mkdir(parents=True)
    for name in cfg.motions:
        (cfg.raw_dir / f"{name}.csv").write_text("原始数据", encoding="utf-8")

    def convert(path: Path, device: str) -> dict[str, np.ndarray]:
        assert path.parent == cfg.raw_dir and path.read_text(encoding="utf-8") == "原始数据"
        return _part(3, float(cfg.motions.index(path.stem)))

    monkeypatch.setitem(tool["main"].__globals__, "convert", convert)
    tool["main"](cfg)
    assert len(list(cfg.out_dir.glob("*.npz"))) == len(list(cfg.raw_dir.glob("*.csv"))) == 2
    with np.load(cfg.merged) as data:
        assert data["motion_ends"].tolist() == [3, 6]
        assert data["joint_pos"][[0, 3], 0].tolist() == [0, 1]

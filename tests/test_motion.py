"""动作文件契约与目标窗口边界测试。

直接消费生成器产物，不在测试中修改受版本控制的夹具。
错误文件写到临时目录；片段末帧必须重复而不能读入下一片段。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nanomimic.motion import validate_motion_file
from nanomimic.specs import Motion


def test_fixture_integrity() -> None:
    directory = Path(__file__).parent / "fixtures"
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["fixture_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected


def test_window_matches_reference() -> None:
    directory = Path(__file__).parent / "fixtures"
    with np.load(directory / "motion.npz") as data, np.load(directory / "reference.npz") as ref:
        motion = Motion(
            *(
                torch.from_numpy(data[name])
                for name in (
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                    "joint_pos",
                    "joint_vel",
                )
            ),
            clip_ends=torch.from_numpy(data["motion_ends"] - 1),
        )
        indices = motion.window_indices(torch.from_numpy(ref["goal_times"]))
        torch.testing.assert_close(indices, torch.from_numpy(ref["goal_indices"]), rtol=0, atol=0)
        assert indices[2].eq(47).all()
        assert indices[5].eq(95).all()


@pytest.mark.parametrize("ends", [[], [96, 48], [0, 96], [48, 48, 96], [95], [48.5, 96]])
def test_invalid_clip_boundaries(tmp_path: Path, ends: list[float]) -> None:
    with np.load(Path(__file__).parent / "fixtures/motion.npz") as source:
        data = dict(source)
    data["motion_ends"] = np.asarray(ends)
    path = tmp_path / "invalid.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="motion_ends"):
        validate_motion_file(path, 29, 30, 0.02)


@pytest.mark.parametrize("field", ["body_pos_w", "body_quat_w", "joint_pos", "fps"])
def test_invalid_motion_values(tmp_path: Path, field: str) -> None:
    with np.load(Path(__file__).parent / "fixtures/motion.npz") as source:
        data = dict(source)
    if field == "fps":
        data[field] = np.array(30)
    elif field == "body_quat_w":
        data[field][:] = 0
    elif field == "joint_pos":
        data[field] = data[field][:, :-1]
    else:
        data[field][0, 0, 0] = np.nan
    path = tmp_path / "invalid.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match=field):
        validate_motion_file(path, 29, 30, 0.02)


def test_single_clip_without_metadata(tmp_path: Path) -> None:
    with np.load(Path(__file__).parent / "fixtures/motion.npz") as source:
        data = dict(source)
    del data["motion_ends"]
    path = tmp_path / "single.npz"
    np.savez(path, **data)
    assert validate_motion_file(path, 29, 30, 0.02).tolist() == [95]

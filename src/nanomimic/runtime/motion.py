"""本地 NPZ 动作文件的边界检查。

文件包含完整机器人 body，进入 Motion 时由仿真器选择跟踪子集。
motion_ends 沿用参考仓库的右开端点；未提供时视为单片段。
"""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def validate_motion_file(path: Path, num_joints: int, num_bodies: int, dt: float) -> Tensor:
    """验证输入并返回 CPU 上的包含式片段末帧索引。"""
    with np.load(path, allow_pickle=False) as data:
        if "joint_pos" not in data or data["joint_pos"].ndim != 2:
            raise ValueError("动作文件的 joint_pos 必须为 [T, J]")
        frames = data["joint_pos"].shape[0]
        if frames == 0:
            raise ValueError("动作文件至少需要一帧")
        shapes = {
            "joint_pos": (frames, num_joints),
            "joint_vel": (frames, num_joints),
            "body_pos_w": (frames, num_bodies, 3),
            "body_quat_w": (frames, num_bodies, 4),
            "body_lin_vel_w": (frames, num_bodies, 3),
            "body_ang_vel_w": (frames, num_bodies, 3),
        }
        for name, shape in shapes.items():
            if name not in data or data[name].shape != shape:
                raise ValueError(f"动作文件的 {name} 必须为 {shape}")
            if not np.isfinite(data[name]).all():
                raise ValueError(f"动作文件的 {name} 含非有限值")
        if not np.allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1, atol=1e-5, rtol=0):
            raise ValueError("动作文件的 body_quat_w 必须为单位四元数")
        if "fps" in data and not np.isclose(float(data["fps"].item()), 1 / dt):
            raise ValueError(f"动作文件的 fps 必须等于控制频率 {1 / dt:g}")
        ends = np.asarray(data.get("motion_ends", [frames]))
        if (
            ends.ndim != 1
            or ends.size == 0
            or not np.issubdtype(ends.dtype, np.integer)
            or ends[0] <= 0
            or ends[-1] != frames
            or np.any(np.diff(ends) <= 0)
        ):
            raise ValueError("motion_ends 必须是严格递增的整数右开端点，最后一个等于总帧数")
        return torch.tensor(ends - 1, dtype=torch.long)

"""纯 CPU 数值测试共享夹具。

机器人和动作只从已生成的夹具读取，不初始化仿真器。
生产算法的预期值始终由独立参考生成，测试不修改夹具。
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from nanomimic.specs import Motion, RobotSpec


@pytest.fixture(scope="session")
def reference() -> dict[str, Tensor]:
    with np.load(Path(__file__).parent / "fixtures/reference.npz") as data:
        return {name: torch.from_numpy(value.copy()) for name, value in data.items()}


@pytest.fixture(scope="session")
def robot(reference: dict[str, Tensor]) -> RobotSpec:
    meta = json.loads((Path(__file__).parent / "fixtures/manifest.json").read_text())
    names = tuple(meta["body_names"])
    return RobotSpec(
        joint_names=tuple(f"joint_{i}" for i in range(29)),
        default_joint_pos=reference["default_joint_pos"],
        soft_joint_pos_limits=reference["soft_joint_pos_limits"],
        action_scale=torch.full((29,), 0.25),
        action_offset=reference["default_joint_pos"],
        dt=0.02,
        body_names=names,
        anchor_body_name=names[0],
        tree_body_names=tuple(str(i) for i in range(30)),
        parents=tuple(reference["robot_parents"].tolist()),
        body_joint=tuple(reference["robot_body_joint"].tolist()),
        body_pos=reference["robot_body_pos"],
        body_quat=reference["robot_body_quat"],
        joint_axis=reference["robot_joint_axis"],
        joint_anchor=reference["robot_joint_anchor"],
        joint_reference=reference["robot_joint_reference"],
        tracked_body_indices=reference["robot_tracked_body_indices"],
        anchor_pos_threshold=float(reference["anchor_pos_threshold"]),
        anchor_ori_threshold=float(reference["anchor_ori_threshold"]),
    )


@pytest.fixture(scope="session")
def motion(robot: RobotSpec) -> Motion:
    with np.load(Path(__file__).parent / "fixtures/motion.npz") as data:
        bodies = [
            torch.from_numpy(data[name])[:, robot.tracked_body_indices]
            for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        ]
        return Motion(
            bodies[0],
            bodies[1],
            bodies[2],
            bodies[3],
            torch.from_numpy(data["joint_pos"]),
            torch.from_numpy(data["joint_vel"]),
            torch.from_numpy(data["motion_ends"]) - 1,
        )

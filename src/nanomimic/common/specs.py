"""仿真与纯张量模块之间的数据契约。

四元数统一为 wxyz，速度与 body 位姿使用世界坐标系。
所有索引遵循 RobotSpec 的关节及 body 顺序；张量不持有仿真器对象。
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class SimConfig:
    """G1 跟踪环境配置，CPU 验收时显式减小 num_envs。"""

    motion_file: Path
    num_envs: int = 2048
    device: str = "cuda:0"
    seed: int = 0


@dataclass(frozen=True)
class Obs:
    """一次观测的独立快照；sensor 不含逐帧均匀噪声。"""

    full_state: Tensor  # [E, 13 + 2J]
    goal: Tensor  # [E, N * (2J + 9)]
    sensor: Tensor  # [E, 6 + 3J]


@dataclass(frozen=True)
class RobotSpec:
    """机器人常量；K 为完整运动树的 body 数，B 为跟踪 body 数。"""

    joint_names: tuple[str, ...]
    default_joint_pos: Tensor  # [J]
    soft_joint_pos_limits: Tensor  # [J, 2]
    action_scale: Tensor  # [J]，原始动作顺序与关节顺序相同
    action_offset: Tensor  # [J]
    dt: float
    body_names: tuple[str, ...]  # B 个跟踪 body
    anchor_body_name: str
    tree_body_names: tuple[str, ...]  # K 个 body，父节点在子节点之前
    parents: tuple[int, ...]  # [K]，根为 -1
    body_joint: tuple[int, ...]  # [K]，无铰链为 -1，否则为关节列
    body_pos: Tensor  # [K, 3]，相对父节点的固定平移
    body_quat: Tensor  # [K, 4]，相对父节点的固定旋转
    joint_axis: Tensor  # [J, 3]，body 局部坐标系
    joint_anchor: Tensor  # [J, 3]，body 局部坐标系
    joint_reference: Tensor  # [J]，MuJoCo 的 qpos0
    tracked_body_indices: Tensor  # [B]，索引完整运动树
    anchor_pos_threshold: float
    anchor_ori_threshold: float


@dataclass(frozen=True)
class ContextSpec:
    """三个随机化通道；各通道可包含多个标量维度。"""

    names: tuple[str, ...]
    widths: tuple[int, ...]
    lower: Tensor  # [C]，依次为摩擦、质心偏移、编码器偏置
    upper: Tensor  # [C]

    def normalize(self, values: Tensor) -> Tensor:
        """把物理量映射到 [-1, 1]，保留越界以便发现契约错误。"""
        return 2 * (values - self.lower) / (self.upper - self.lower) - 1


@dataclass(frozen=True)
class Motion:
    """预计算的动作张量；clip_ends 存各片段最后一帧的包含式索引。"""

    body_pos_w: Tensor  # [T, B, 3]
    body_quat_w: Tensor  # [T, B, 4]
    body_lin_vel_w: Tensor  # [T, B, 3]
    body_ang_vel_w: Tensor  # [T, B, 3]
    joint_pos: Tensor  # [T, J]
    joint_vel: Tensor  # [T, J]
    clip_ends: Tensor  # [片段数]

    @property
    def num_frames(self) -> int:
        return self.joint_pos.shape[0]

    def last_frame(self, time: Tensor) -> Tensor:
        """返回每个当前帧所属片段的末帧。"""
        return self.clip_ends[torch.searchsorted(self.clip_ends, time)]

    def window_indices(self, time: Tensor) -> Tensor:
        """返回 [E, N=10]，间隔 5 步且不跨片段的目标索引。"""
        offsets = torch.arange(10, device=time.device) * 5
        return torch.minimum(time[:, None] + offsets, self.last_frame(time)[:, None])

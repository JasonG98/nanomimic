"""采集和想象共用的传感器合成、均匀噪声与历史窗口。

proprio 顺序固定为重力、角速度、相对关节位置、关节速度、上一动作。
历史为 [B, H, D]、最旧帧在前；重置填充同一份已采样噪声的帧。
"""

import torch
from torch import Tensor

from nanomimic.common.rotation import inverse_rotate
from nanomimic.models.model import joint_state


def synth_proprio(state: Tensor, aux: Tensor, last_action: Tensor, default_joint_pos: Tensor) -> Tensor:
    """显式传入默认姿态，避免从全局机器人或仿真器读取常量。"""
    position, velocity = joint_state(state).chunk(2, dim=-1)
    gravity = inverse_rotate(aux[..., 3:7], aux.new_tensor([0, 0, -1]))
    return torch.cat((gravity, state[..., 7:10], position - default_joint_pos, velocity, last_action), dim=-1)


def noise_half_width(frame: Tensor) -> Tensor:
    joints = (frame.shape[-1] - 6) // 3
    return frame.new_tensor([0.05] * 3 + [0.2] * 3 + [0.01] * joints + [0.5] * joints + [0.0] * joints)


def add_noise(frame: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """唯一的逐帧噪声入口；动作通道严格不加噪声。"""
    uniform = torch.rand(frame.shape, device=frame.device, dtype=frame.dtype, generator=generator)
    return frame + (2 * uniform - 1) * noise_half_width(frame)


def push_history(history: Tensor, frame: Tensor) -> Tensor:
    """函数式更新，保留想象反向传播所需的旧张量。"""
    return torch.cat((history[:, 1:], frame[:, None]), dim=1)


class HistoryRing:
    """小型帧历史；首次 push 相当于全体行重置。"""

    def __init__(self, length: int) -> None:
        if length < 1:
            raise ValueError("历史长度必须大于零")
        self.length = length
        self._frames: Tensor | None = None

    def push(self, frame: Tensor) -> None:
        self._frames = (
            frame[:, None].expand(-1, self.length, -1).clone()
            if self._frames is None
            else push_history(self._frames, frame)
        )

    def reset(self, rows: Tensor, frame: Tensor) -> None:
        """rows 为布尔掩码或行索引，frame 只含选中行的已加噪观测。"""
        updated = self.window().clone()
        updated[rows] = frame[:, None]
        self._frames = updated

    def window(self) -> Tensor:
        if self._frames is None:
            raise RuntimeError("历史尚未初始化，请先 push 一帧")
        return self._frames

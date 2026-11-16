"""从本地动作库构造世界模型与 Actor 的目标窗口。

窗口固定十帧、间隔五控制步，索引在所属 clip 末帧钳位。
相对朝向采用旋转矩阵前两列按行展开，保持与部署契约一致。
"""

import torch
from torch import Tensor

from nanomimic.rotation import conjugate, inverse_rotate, matrix, multiply
from nanomimic.specs import Motion


def wm_goal(motion: Motion, aux: Tensor, time: Tensor) -> Tensor:
    indices = motion.window_indices(time)
    quat = aux[:, None, 3:7]
    position = inverse_rotate(quat, motion.body_pos_w[indices, 0] - aux[:, None, :3])
    orientation = matrix(multiply(conjugate(quat), motion.body_quat_w[indices, 0]))[..., :2]
    content = torch.cat((motion.joint_pos[indices], motion.joint_vel[indices]), dim=-1)
    return torch.cat((content.flatten(1), position.flatten(1), orientation.flatten(1)), dim=-1)


def actor_goal(goal: Tensor) -> Tensor:
    """仅移除 10 × 3 的相对位置，保留 content 与 10 × 6 的相对朝向。"""
    return torch.cat((goal[..., :-90], goal[..., -60:]), dim=-1)

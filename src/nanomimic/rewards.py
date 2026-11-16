"""世界系跟踪奖励和可微软终止。

目标位姿按预测根的 xy 与相对偏航重锚定，速度仍直接比较世界系值。
正奖励上界为 5 × dt；关节越限使用软限位，倾斜使用投影重力 z 分量。
"""

import torch
from torch import Tensor

from nanomimic.model import Bodies, joint_state
from nanomimic.rotation import conjugate, matrix, multiply, rotate
from nanomimic.specs import RobotSpec


def _angle_squared(a: Tensor, b: Tensor) -> Tensor:
    cosine = ((a * b).sum((-2, -1)) - 1) / 2
    return cosine.clamp(-1 + 1e-7, 1 - 1e-7).acos().square()


def reanchor(target: Bodies, aux: Tensor) -> tuple[Tensor, Tensor]:
    """返回重锚定后的位置和旋转矩阵。"""
    difference = multiply(aux[..., 3:7], conjugate(target.quat[..., 0, :]))
    w, x, y, z = difference.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    turn = torch.stack(((yaw / 2).cos(), zero, zero, (yaw / 2).sin()), dim=-1)
    turn = turn / turn.norm(dim=-1, keepdim=True)
    origin = torch.cat((aux[..., :2], target.pos[..., 0, 2:3]), dim=-1)
    position = origin[..., None, :] + rotate(turn[..., None, :], target.pos - target.pos[..., :1, :])
    orientation = matrix(turn)[..., None, :, :] @ matrix(target.quat)
    return position, orientation


def tracking_reward(
    aux: Tensor, state: Tensor, predicted: Bodies, target: Bodies, action: Tensor, last_action: Tensor, robot: RobotSpec
) -> Tensor:
    target_pos, target_rot = reanchor(target, aux)
    pred_rot = matrix(predicted.quat)
    terms = (
        ((predicted.pos[..., 0, :] - target.pos[..., 0, :]).square().sum(-1), 0.5, 0.3),
        (_angle_squared(pred_rot[..., 0, :, :], matrix(target.quat[..., 0, :])), 0.5, 0.4),
        ((predicted.pos - target_pos).square().sum(-1).mean(-1), 1.0, 0.3),
        (_angle_squared(pred_rot, target_rot).mean(-1), 1.0, 0.4),
        ((predicted.linear - target.linear).square().sum(-1).mean(-1), 1.0, 1.0),
        ((predicted.angular - target.angular).square().sum(-1).mean(-1), 1.0, 3.14),
    )
    positive = torch.stack([weight * (-error / std**2).exp() for error, weight, std in terms]).sum(0)
    position, _ = joint_state(state).chunk(2, dim=-1)
    lo, hi = robot.soft_joint_pos_limits.unbind(-1)
    limit = ((lo - position).clamp_min(0) + (position - hi).clamp_min(0)).sum(-1)
    rate = (action - last_action).square().sum(-1)
    return robot.dt * (positive - 0.1 * rate - 2 * limit)


def termination_probability(aux: Tensor, target: Bodies, robot: RobotSpec) -> Tensor:
    index = robot.body_names.index(robot.anchor_body_name)
    height = (aux[..., 2] - target.pos[..., index, 2]).abs()
    gravity_z = 2 * aux[..., 4:6].square().sum(-1) - 1
    target_z = 2 * target.quat[..., index, 1:3].square().sum(-1) - 1
    error = torch.maximum(
        height / robot.anchor_pos_threshold, (gravity_z - target_z).abs() / robot.anchor_ori_threshold
    )
    return ((error - 1) / 0.05).sigmoid()

"""纯张量四元数运算，供积分、FK 与目标构造共用。

四元数顺序为 wxyz，乘积表示旋转复合；输入允许批维广播。
轴角在零角度附近保持有限梯度，不依赖仿真器数学库。
"""

import torch
from torch import Tensor


def conjugate(q: Tensor) -> Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def multiply(a: Tensor, b: Tensor) -> Tensor:
    av, bv = torch.broadcast_tensors(a[..., 1:], b[..., 1:])
    scalar = a[..., :1] * b[..., :1] - (av * bv).sum(-1, keepdim=True)
    vector = a[..., :1] * bv + b[..., :1] * av + torch.linalg.cross(av, bv)
    return torch.cat((scalar, vector), dim=-1)


def rotate(q: Tensor, v: Tensor) -> Tensor:
    axis, v = torch.broadcast_tensors(q[..., 1:], v)
    cross = 2 * torch.linalg.cross(axis, v)
    return v + q[..., :1] * cross + torch.linalg.cross(axis, cross)


def inverse_rotate(q: Tensor, v: Tensor) -> Tensor:
    return rotate(conjugate(q), v)


def from_rotvec(vector: Tensor) -> Tensor:
    """旋转向量转单位四元数；正则化的长度避免零点 sqrt 导数奇异。"""
    angle = (vector.square().sum(-1, keepdim=True) + 1e-8).sqrt()
    q = torch.cat(((angle / 2).cos(), vector * (angle / 2).sin() / angle), dim=-1)
    return q / q.norm(dim=-1, keepdim=True)


def matrix(q: Tensor) -> Tensor:
    """返回 [..., 3, 3]，可接受单位长度附近的舍入误差。"""
    w, x, y, z = q.unbind(-1)
    scale = 2 / q.square().sum(-1)
    rows = (
        1 - scale * (y * y + z * z),
        scale * (x * y - w * z),
        scale * (x * z + w * y),
        scale * (x * y + w * z),
        1 - scale * (x * x + z * z),
        scale * (y * z - w * x),
        scale * (x * z - w * y),
        scale * (y * z + w * x),
        1 - scale * (x * x + y * y),
    )
    return torch.stack(rows, dim=-1).unflatten(-1, (3, 3))

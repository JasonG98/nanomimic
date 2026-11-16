"""广义坐标世界模型、半隐式积分和可微运动树。

full_state 的根速度在世界系，rep_state 的根速度在机体系；四元数为 wxyz。
网络只预测速度增量；根位姿与所有 body 的位姿、速度由纯函数推导。
"""

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from nanomimic.networks import RunningNorm, mlp
from nanomimic.rotation import from_rotvec, inverse_rotate, matrix, multiply, rotate
from nanomimic.specs import RobotSpec


def represent(full_state: Tensor) -> Tensor:
    """把 [..., 13+2J] 转成 [..., 10+2J]。"""
    joints = (full_state.shape[-1] - 13) // 2
    pos, quat = full_state[..., :3], full_state[..., 3:7]
    linear = full_state[..., 7 + joints : 10 + joints]
    angular = full_state[..., 10 + joints : 13 + joints]
    down = full_state.new_tensor([0, 0, -1])
    return torch.cat(
        (
            pos[..., 2:3],
            inverse_rotate(quat, down),
            inverse_rotate(quat, linear),
            inverse_rotate(quat, angular),
            full_state[..., 7 : 7 + joints],
            full_state[..., 13 + joints :],
        ),
        dim=-1,
    )


def aux_state(full_state: Tensor) -> Tensor:
    return full_state[..., :7]


def joint_state(rep_state: Tensor) -> Tensor:
    return rep_state[..., 10:]


def integrate(aux: Tensor, state: Tensor, prediction: Tensor, dt: float) -> tuple[Tensor, Tensor]:
    """返回 (next_aux, next_rep)，增量不再乘 dt；位姿使用完整控制步。"""
    qj, vj = joint_state(state).chunk(2, dim=-1)
    quat = aux[..., 3:7]
    linear = rotate(quat, state[..., 4:7] + prediction[..., :3])
    angular = rotate(quat, state[..., 7:10] + prediction[..., 3:6])
    velocity = vj + prediction[..., 6:]
    position = aux[..., :3] + dt * linear
    orientation = multiply(from_rotvec(dt * angular), quat)
    orientation = orientation / orientation.norm(dim=-1, keepdim=True)
    next_aux = torch.cat((position, orientation), dim=-1)
    full = torch.cat((next_aux, qj + dt * velocity, linear, angular, velocity), dim=-1)
    return next_aux, represent(full)


def reconstruct_full_state(aux: Tensor, state: Tensor) -> Tensor:
    qj, vj = joint_state(state).chunk(2, dim=-1)
    quat = aux[..., 3:7]
    return torch.cat(
        (aux[..., :2], state[..., :1], quat, qj, rotate(quat, state[..., 4:7]), rotate(quat, state[..., 7:10]), vj),
        dim=-1,
    )


@dataclass(frozen=True)
class Bodies:
    """世界系的跟踪 body；K 为跟踪 body 数，速度在线框原点处。"""

    pos: Tensor  # [B, K, 3]
    quat: Tensor  # [B, K, 4]
    linear: Tensor  # [B, K, 3]
    angular: Tensor  # [B, K, 3]


def world_bodies(aux: Tensor, state: Tensor, robot: RobotSpec) -> Bodies:
    """按拓扑序传播刚体变换与扭量，包含铰链原点偏移。"""
    positions, orientations = [aux[..., :3]], [aux[..., 3:7]]
    linear = [rotate(orientations[0], state[..., 4:7])]
    angular = [rotate(orientations[0], state[..., 7:10])]
    qj, vj = joint_state(state).chunk(2, dim=-1)
    for body in range(1, len(robot.parents)):
        parent, joint = robot.parents[body], robot.body_joint[body]
        base_pos = positions[parent] + rotate(orientations[parent], robot.body_pos[body])
        base_quat = multiply(orientations[parent], robot.body_quat[body])
        spin = angular[parent]
        if joint >= 0:
            axis = robot.joint_axis[joint]
            angle = qj[..., joint : joint + 1] - robot.joint_reference[joint]
            turn = torch.cat(((angle / 2).cos(), (angle / 2).sin() * axis), dim=-1)
            quat = multiply(base_quat, turn)
            pivot = base_pos + rotate(base_quat, robot.joint_anchor[joint])
            pos = pivot - rotate(quat, robot.joint_anchor[joint])
            spin = spin + rotate(base_quat, axis) * vj[..., joint : joint + 1]
            speed = (
                linear[parent]
                + torch.linalg.cross(angular[parent], pivot - positions[parent])
                + torch.linalg.cross(spin, pos - pivot)
            )
        else:
            pos, quat = base_pos, base_quat
            speed = linear[parent] + torch.linalg.cross(spin, pos - positions[parent])
        positions.append(pos)
        orientations.append(quat)
        linear.append(speed)
        angular.append(spin)
    indices = robot.tracked_body_indices
    return Bodies(
        *(
            torch.stack(values, dim=-2).index_select(-2, indices)
            for values in (positions, orientations, linear, angular)
        )
    )


def loss_groups(state: Tensor, aux: Tensor) -> dict[str, Tensor]:
    qj, vj = joint_state(state).chunk(2, dim=-1)
    return {
        "root_pos": aux[..., :3],
        "root_ori": matrix(aux[..., 3:7]).flatten(-2) * sqrt(0.5),
        "joint_pos": qj,
        "joint_vel": vj,
    }


def bryson_weights() -> dict[str, float]:
    return {
        name: 1 / nominal**2
        for name, nominal in (("root_pos", 0.01), ("root_ori", 0.02), ("joint_pos", 0.1), ("joint_vel", 3.14))
    }


@dataclass(frozen=True)
class DynamicsConfig:
    """默认训练宽度保持规格值，CPU 验收可显式减小。"""

    hidden: tuple[int, ...] = (2048, 2048, 2048)


class Dynamics(nn.Module):
    """原始动作先转成关节目标，再归一化；前向不更新统计量。"""

    action_scale: Tensor
    action_offset: Tensor

    def __init__(self, robot: RobotSpec, ctx_dim: int, cfg: DynamicsConfig = DynamicsConfig()) -> None:
        super().__init__()
        joints = len(robot.joint_names)
        self.dt = robot.dt
        self.register_buffer("action_scale", robot.action_scale.clone())
        self.register_buffer("action_offset", robot.action_offset.clone())
        self.state_norm = RunningNorm(10 + 2 * joints)
        self.action_norm = RunningNorm(joints)
        self.mlp = mlp(10 + 3 * joints + ctx_dim, cfg.hidden, 6 + joints)

    def process_action(self, action: Tensor) -> Tensor:
        return action * self.action_scale + self.action_offset

    def forward(self, state: Tensor, action: Tensor, aux: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor]:
        inputs = torch.cat((self.state_norm(state), self.action_norm(self.process_action(action)), ctx), dim=-1)
        return integrate(aux, state, self.mlp(inputs), self.dt)

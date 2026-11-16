"""T3 对拍与可微性验收。

所有对拍固定使用 fp32、绝对误差 1e-5、相对误差零。
额外覆盖零角速度与在线统计量，避免训练时才发现奇异梯度。
"""

import torch
from torch import Tensor

from nanomimic.model import (
    Dynamics,
    DynamicsConfig,
    aux_state,
    bryson_weights,
    integrate,
    joint_state,
    loss_groups,
    reconstruct_full_state,
    represent,
    world_bodies,
)
from nanomimic.networks import RunningNorm
from nanomimic.specs import RobotSpec


def test_state_and_integration(reference: dict[str, Tensor]) -> None:
    full = reference["full_state"]
    aux, state = aux_state(full), represent(full)
    results = {
        "aux_state": aux,
        "rep_state": state,
        "joint_state": joint_state(state),
        "reconstructed": reconstruct_full_state(aux, state),
    }
    results["next_aux_state"], results["next_rep_state"] = integrate(aux, state, reference["prediction"], 0.02)
    results.update({f"loss_{name}": value for name, value in loss_groups(state, aux).items()})
    results["bryson_weights"] = torch.tensor(list(bryson_weights().values()))
    for name, actual in results.items():
        torch.testing.assert_close(actual, reference[name], atol=1e-5, rtol=0, msg=name)


def test_fk(reference: dict[str, Tensor], robot: RobotSpec) -> None:
    full = reference["full_state"]
    bodies = world_bodies(aux_state(full), represent(full), robot)
    for field, name in (
        (bodies.pos, "body_pos_w"),
        (bodies.quat, "body_quat_w"),
        (bodies.linear, "body_lin_vel_w"),
        (bodies.angular, "body_ang_vel_w"),
    ):
        torch.testing.assert_close(field, reference[name], atol=1e-5, rtol=0, msg=name)


def test_zero_rotation_gradients(reference: dict[str, Tensor], robot: RobotSpec) -> None:
    state = reference["rep_state"][:2].clone()
    state[:, 7:10] = 0
    prediction = torch.zeros(2, 35, requires_grad=True)
    aux, state = integrate(reference["aux_state"][:2], state, prediction, robot.dt)
    bodies = world_bodies(aux, state, robot)
    loss = bodies.pos.square().sum() + bodies.quat.sum() + bodies.linear.sum() + bodies.angular.sum()
    (gradient,) = torch.autograd.grad(loss, prediction)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_running_norm_and_dynamics(reference: dict[str, Tensor], robot: RobotSpec) -> None:
    norm = RunningNorm(3)
    data = torch.randn(100, 3) * 3 + 4
    norm.update(data[:17])
    norm.update(data[17:])
    torch.testing.assert_close(norm.mean, data.mean(0))
    torch.testing.assert_close(norm.var, data.var(0, correction=0))
    count = norm.count.clone()
    norm(data)
    assert torch.equal(count, norm.count)
    model = Dynamics(robot, 33, DynamicsConfig((16, 16, 16)))
    action = reference["action"].clone().requires_grad_()
    aux, state = model(reference["rep_state"], action, reference["aux_state"], torch.zeros(32, 33))
    (aux.square().sum() + state.square().sum()).backward()
    assert action.grad is not None and torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
    torch.testing.assert_close(model.process_action(action), action * robot.action_scale + robot.action_offset)

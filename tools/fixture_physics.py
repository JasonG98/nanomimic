"""用 MuJoCo CPU 计算夹具的独立 FK 真值。

不调用 nanomimic 的数值实现；线速度取 body 原点而非惯性质心。
合成动作仅供契约验收，不能替代训练动作或效果验收。
"""

import numpy as np
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity

# MuJoCo 3.11 的公开 C 扩展导出未提供可被 pyright 解析的类型桩。
from mujoco import MjData, MjModel, mj_forward, mj_jac, mju_quat2Mat  # pyright: ignore[reportAttributeAccessIssue]
from numpy.typing import NDArray


def robot_model() -> MjModel:
    """编译发布版 G1 资产，包含默认关节姿态。"""
    return Entity(get_g1_robot_cfg()).spec.compile()


def body_states(model: MjModel, full_state: NDArray[np.float32]) -> dict[str, NDArray[np.float32]]:
    """从 [T, 13+2J] 世界系状态计算全部机器人 body 的位姿与速度。"""
    frames, joints = full_state.shape[0], model.nv - 6
    result = {
        name: np.empty((frames, model.nbody - 1, width), dtype=np.float32)
        for name, width in (("body_pos_w", 3), ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3))
    }
    data = MjData(model)
    jac_pos, jac_rot = np.empty((3, model.nv)), np.empty((3, model.nv))
    rotation = np.empty(9)
    for frame, state in enumerate(full_state):
        data.qpos[:] = state[: 7 + joints]
        data.qvel[:3] = state[7 + joints : 10 + joints]
        mju_quat2Mat(rotation, np.asarray(state[3:7], dtype=np.float64))
        # free joint 的角速度在局部坐标系；full_state 中则为世界系。
        data.qvel[3:6] = rotation.reshape(3, 3).T @ state[10 + joints : 13 + joints]
        data.qvel[6:] = state[13 + joints :]
        mj_forward(model, data)
        result["body_pos_w"][frame] = data.xpos[1:]
        result["body_quat_w"][frame] = data.xquat[1:]
        for body in range(1, model.nbody):
            mj_jac(model, data, jac_pos, jac_rot, data.xpos[body], body)
            result["body_lin_vel_w"][frame, body - 1] = jac_pos @ data.qvel
            result["body_ang_vel_w"][frame, body - 1] = jac_rot @ data.qvel
    return result


def synthetic_motion(model: MjModel) -> dict[str, NDArray]:
    """生成两个 48 帧片段，包含可识别的边界及非零关节速度。"""
    frames, joints = 96, model.nv - 6
    state = np.zeros((frames, 13 + 2 * joints), dtype=np.float32)
    state[:, : 7 + joints] = model.key_qpos[0]
    t = (np.arange(frames) % 48)[:, None] * 0.02
    phase = np.linspace(0, np.pi, joints)[None]
    state[:, 7 : 7 + joints] += 0.02 * np.sin(t + phase)
    state[:, 13 + joints :] = 0.02 * np.cos(t + phase)
    state[48:, 0] = 0.3
    return {
        **body_states(model, state),
        "joint_pos": state[:, 7 : 7 + joints],
        "joint_vel": state[:, 13 + joints :],
        "fps": np.array(50, dtype=np.float32),
        "motion_ends": np.array([48, 96], dtype=np.int64),
    }

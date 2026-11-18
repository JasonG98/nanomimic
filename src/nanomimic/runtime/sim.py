"""mjlab 发布版的 G1 跟踪适配层。

只有此边界接触仿真器；对外传递 dataclass 与独立张量快照。
环境自动重置，done 行的返回观测属于新回合；动作时间在片段末帧钳住。
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlBaseRunnerCfg
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms
from torch import Tensor

from nanomimic.common.specs import ContextSpec, Motion, Obs, RobotSpec, SimConfig
from nanomimic.runtime.motion import validate_motion_file


class _ClipMotion(MotionCommand):
    """沿用 adaptive 重置，补齐发布版缺少的片段末帧钳位。"""

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv) -> None:
        robot = env.scene[cfg.entity_name]
        ends = validate_motion_file(Path(cfg.motion_file), robot.num_joints, len(robot.body_names), env.step_dt)
        super().__init__(cfg, env)
        source = self.motion
        self.reference = Motion(
            source.body_pos_w,
            source.body_quat_w,
            source.body_lin_vel_w,
            source.body_ang_vel_w,
            source.joint_pos,
            source.joint_vel,
            ends.to(self.device),
        )

    def _update_command(self, env_ids: Tensor | None = None) -> None:
        rows = slice(None) if env_ids is None else env_ids
        self.time_steps[rows] = torch.minimum(
            self.time_steps[rows] + 1, self.reference.last_frame(self.time_steps[rows])
        )
        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()
        self.update_relative_body_poses()
        if env_ids is None and self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count.lerp_(self._current_bin_failed, self.cfg.adaptive_alpha)
            self._current_bin_failed.zero_()


class _ClipMotionCfg(MotionCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> _ClipMotion:
        return _ClipMotion(self, env)


def _full_state(env: ManagerBasedRlEnv) -> Tensor:
    data = env.scene["robot"].data
    return torch.cat(
        (
            data.root_link_pos_w - env.scene.env_origins,
            data.root_link_quat_w,
            data.joint_pos,
            data.root_link_lin_vel_w,
            data.root_link_ang_vel_w,
            data.joint_vel,
        ),
        dim=-1,
    )


def _goal(env: ManagerBasedRlEnv) -> Tensor:
    command = cast(_ClipMotion, env.command_manager.get_term("motion"))
    motion = command.reference
    idx = motion.window_indices(command.time_steps)
    root_pos = command.robot_anchor_pos_w - env.scene.env_origins
    root_quat = command.robot_anchor_quat_w
    pos, quat = subtract_frame_transforms(
        root_pos[:, None].expand(-1, 10, -1),
        root_quat[:, None].expand(-1, 10, -1),
        motion.body_pos_w[idx, 0],
        motion.body_quat_w[idx, 0],
    )
    # rot6d 为旋转矩阵前两列按行展开，与参考部署契约一致。
    orientation = matrix_from_quat(quat)[..., :2].flatten(-2)
    content = torch.cat((motion.joint_pos[idx], motion.joint_vel[idx]), dim=-1)
    return torch.cat((content.flatten(1), pos.flatten(1), orientation.flatten(1)), dim=-1)


def _env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_flat_tracking_env_cfg()
    cfg.scene.num_envs = SimConfig.num_envs
    source = cast(MotionCommandCfg, cfg.commands["motion"])
    command = _ClipMotionCfg(**{field.name: getattr(source, field.name) for field in fields(source)})
    command.anchor_body_name = "pelvis"
    command.sampling_mode = "adaptive"
    cfg.commands["motion"] = command
    cfg.observations = {
        "full_state": ObservationGroupCfg(
            terms={"full_state": ObservationTermCfg(func=_full_state)},
            enable_corruption=False,
        ),
        "goal": ObservationGroupCfg(terms={"goal": ObservationTermCfg(func=_goal)}, enable_corruption=False),
        "actor": ObservationGroupCfg(
            terms={
                "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
                "base_ang_vel": ObservationTermCfg(
                    func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
                ),
                "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"biased": True}),
                "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                "actions": ObservationTermCfg(func=mdp.last_action),
            },
            enable_corruption=False,
        ),
    }
    cfg.events = {key: cfg.events[key] for key in ("foot_friction", "base_com", "encoder_bias")}
    cfg.rewards.pop("self_collisions")
    cfg.rewards["joint_limit"].weight = -2.0
    cfg.terminations.pop("ee_body_pos")
    return cfg


def _action_affine(env: ManagerBasedRlEnv) -> tuple[Tensor, Tensor]:
    robot = env.scene["robot"]
    action = cast(JointPositionAction, env.action_manager.get_term("joint_pos"))
    if action.target_names != list(robot.joint_names):
        raise ValueError("动作顺序必须与机器人关节顺序一致")

    def affine(value: Tensor | float) -> Tensor:
        tensor = torch.as_tensor(value, device=env.device, dtype=torch.float32)
        return torch.broadcast_to(tensor, (env.num_envs, robot.num_joints))[0].clone()

    return affine(action.scale), affine(action.offset)


def deployment_metadata() -> tuple[int, dict[str, str]]:
    """从无动作指令的 CPU 场景解析关节顺序、姿态和动作参数。"""
    cfg = _env_cfg()
    command = cast(MotionCommandCfg, cfg.commands["motion"])
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(
        ManagerBasedRlEnvCfg(
            decimation=cfg.decimation,
            scene=cfg.scene,
            sim=cfg.sim,
            actions=cfg.actions,
            observations={"actor": cfg.observations["actor"]},
            events={},
        ),
        device="cpu",
    )
    try:
        robot = env.scene["robot"]
        scale, offset = _action_affine(env)
        base = get_base_metadata(env, "")
        fields = {key: base[key] for key in ("joint_names", "joint_stiffness", "joint_damping", "default_joint_pos")}
        fields.update(
            action_scale=scale.tolist(),
            action_offset=offset.tolist(),
            control_dt=env.step_dt,
            motion_body_names=list(robot.body_names),
        )
        metadata = {key: json.dumps(value, allow_nan=False) for key, value in fields.items()}
        metadata["anchor_body_name"] = command.anchor_body_name
        return robot.num_joints, metadata
    finally:
        env.close()


def _robot_spec(env: ManagerBasedRlEnv) -> RobotSpec:
    robot = env.scene["robot"]
    model = env.sim.mj_model
    body_ids = robot.indexing.body_ids.cpu().tolist()
    joint_ids = robot.indexing.joint_ids.cpu().tolist()
    parent_map = {body: row for row, body in enumerate(body_ids)}
    joint_map = {int(model.jnt_bodyid[joint]): col for col, joint in enumerate(joint_ids)}
    command = cast(MotionCommand, env.command_manager.get_term("motion"))
    scale, offset = _action_affine(env)

    def tensor(value: object) -> Tensor:
        return torch.as_tensor(value, device=env.device, dtype=torch.float32).clone()

    return RobotSpec(
        joint_names=tuple(robot.joint_names),
        default_joint_pos=robot.data.default_joint_pos[0].clone(),
        soft_joint_pos_limits=robot.data.soft_joint_pos_limits[0].clone(),
        action_scale=scale,
        action_offset=offset,
        dt=env.step_dt,
        body_names=tuple(command.cfg.body_names),
        anchor_body_name=command.cfg.anchor_body_name,
        tree_body_names=tuple(robot.body_names),
        parents=tuple(parent_map.get(int(model.body_parentid[body]), -1) for body in body_ids),
        body_joint=tuple(joint_map.get(body, -1) for body in body_ids),
        body_pos=tensor(model.body_pos[body_ids]),
        body_quat=tensor(model.body_quat[body_ids]),
        joint_axis=tensor(model.jnt_axis[joint_ids]),
        joint_anchor=tensor(model.jnt_pos[joint_ids]),
        joint_reference=tensor(model.qpos0[model.jnt_qposadr[joint_ids]]),
        tracked_body_indices=command.body_indexes.clone(),
        anchor_pos_threshold=env.cfg.terminations["anchor_pos"].params["threshold"],
        anchor_ori_threshold=env.cfg.terminations["anchor_ori"].params["threshold"],
    )


class Sim:
    """G1 仿真资源，使用 with 或 close() 释放；不添加传感器噪声。"""

    def __init__(self, cfg: SimConfig) -> None:
        if cfg.num_envs < 1:
            raise ValueError("num_envs 必须大于零")
        if not cfg.motion_file.is_file():
            raise FileNotFoundError(f"找不到本地动作文件：{cfg.motion_file}")
        env_cfg = _env_cfg()
        task_id = "Nanomimic-Tracking-Flat-Unitree-G1"
        if task_id not in list_tasks():
            register_mjlab_task(task_id, _env_cfg(), _env_cfg(), RslRlBaseRunnerCfg())
        env_cfg.scene.num_envs = cfg.num_envs
        env_cfg.seed = cfg.seed
        cast(MotionCommandCfg, env_cfg.commands["motion"]).motion_file = str(cfg.motion_file.resolve())
        self._env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
        try:
            self.robot = _robot_spec(self._env)
            self._command = cast(_ClipMotion, self._env.command_manager.get_term("motion"))
            self.motion = self._command.reference
            self._bind_context()
        except Exception:
            self._env.close()
            raise

    def _bind_context(self) -> None:
        env, robot = self._env, self._env.scene["robot"]
        events = env.cfg.events
        friction = events["foot_friction"].params
        com = events["base_com"].params
        foot = robot.find_geoms(friction["asset_cfg"].geom_names)[0][0]
        torso = robot.find_bodies(com["asset_cfg"].body_names)[0][0]
        self._foot_id = int(robot.indexing.geom_ids[foot].item())
        self._com_id = int(robot.indexing.body_ids[torso].item())
        # 事件采用 add；先减去名义惯性质心，才是范围内的随机偏移量。
        self._nominal_com = torch.tensor(
            env.sim.mj_model.body_ipos[self._com_id], device=env.device, dtype=torch.float32
        )
        ranges = [friction["ranges"], *[com["ranges"][axis] for axis in range(3)]]
        ranges += [events["encoder_bias"].params["bias_range"]] * robot.num_joints
        bounds = torch.tensor(ranges, device=env.device, dtype=torch.float32)
        self.ctx_spec = ContextSpec(
            ("foot_friction", "base_com", "encoder_bias"),
            (1, 3, robot.num_joints),
            bounds[:, 0],
            bounds[:, 1],
        )

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> torch.device:
        return torch.device(self._env.device)

    @property
    def motion_time(self) -> Tensor:
        return self._command.time_steps.clone()

    def read_ctx(self) -> Tensor:
        """读取实际模型随机化参数并归一化为 [E, C]。"""
        model = self._env.sim.model
        raw = torch.cat(
            (
                model.geom_friction[:, self._foot_id, 0:1],
                model.body_ipos[:, self._com_id] - self._nominal_com,
                self._env.scene["robot"].data.encoder_bias,
            ),
            dim=-1,
        )
        return self.ctx_spec.normalize(raw)

    @staticmethod
    def _snapshot(obs: dict[str, Tensor | dict[str, Tensor]]) -> Obs:
        values = [obs[name] for name in ("full_state", "goal", "actor")]
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError("观测组必须拼接为张量")
        return Obs(*(cast(Tensor, value).clone() for value in values))

    def reset(self) -> Obs:
        """重置全部环境；重复调用会重新采样动作起点。"""
        obs, _ = self._env.reset()
        return self._snapshot(obs)

    def step(self, action: Tensor) -> tuple[Obs, Tensor, Tensor, Tensor]:
        """推进一控制步；done 行已自动重置，不能用作旧回合的后继状态。"""
        if action.shape != (self.num_envs, len(self.robot.joint_names)):
            raise ValueError("action 必须为 [num_envs, 29]")
        if not torch.isfinite(action).all():
            raise ValueError("action 必须全部为有限值")
        obs, reward, terminated, truncated, _ = self._env.step(action)
        return self._snapshot(obs), reward.clone(), terminated.clone(), truncated.clone()

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> Sim:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

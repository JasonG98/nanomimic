"""调用只读参考仓库生成 T2、T3、T5 的 CPU 对拍夹具。

输入由固定随机种子产生；期望值来自 SuperMimic 或 MuJoCo。
产物附带版本与源码哈希，禁止手工修改 tests/fixtures 下的文件。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import tyro
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_euler_xyz

from fixture_physics import body_states, robot_model, synthetic_motion


@dataclass(frozen=True)
class Config:
    """参考仓库路径、输出目录与固定随机种子。"""

    reference: Path
    out: Path = Path("tests/fixtures")
    seed: int = 20261116


def generate(cfg: Config) -> None:
    """生成可供后续纯张量模块直接加载的 NPZ 与来源清单。"""
    reference = cfg.reference.resolve()
    if not (reference / "src/supermimic/representations.py").is_file():
        raise ValueError(f"找不到参考仓库：{reference}")
    # 参考仓库只读，包括不写入 Python 字节码缓存。
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference / "src"))
    representations = importlib.import_module("supermimic.representations")
    reward_module = importlib.import_module("supermimic.utils.tracking_reward")
    world_env = importlib.import_module("supermimic.envs.world_model_env")
    task_module = importlib.import_module("supermimic.tasks.env_cfgs")
    learner = importlib.import_module("supermimic.learners.wm_learner")
    cfg.out.mkdir(parents=True, exist_ok=True)
    model = robot_model()
    motion = synthetic_motion(model)
    np.savez_compressed(cfg.out / "motion.npz", allow_pickle=False, **motion)
    torch.manual_seed(cfg.seed)
    samples, joints, dt = 32, model.nv - 6, 0.02
    full = torch.randn(samples, 13 + 2 * joints) * 0.3
    full[:, 2] += 0.8
    angles = torch.randn(samples, 3) * 0.7
    angles[0] = 0
    full[:, 3:7] = quat_from_euler_xyz(*angles.unbind(-1))
    default = torch.tensor(model.key_qpos[0, 7:], dtype=torch.float32)
    full[:, 7 : 7 + joints] += default
    rep = representations.GenCoordRepresentation(joints, default)
    rep.substep_coeff = 1.0  # SPEC 要求控制步半隐式积分，不使用参考工厂的 0.625。
    aux = rep.extract_aux_state(full)
    state = rep.represent(full)
    prediction = torch.randn(samples, 6 + joints) * 0.2
    prediction[0] = 0
    next_aux, next_state = rep.apply_prediction(aux, state, prediction, dt)
    env_cfg = task_module.gencoord_env_cfg()
    names = env_cfg.commands["motion"].body_names
    all_names = [model.body(body).name for body in range(1, model.nbody)]
    tracked = [all_names.index(name) for name in names]
    bodies = {key: torch.from_numpy(value[:, tracked]) for key, value in body_states(model, full.numpy()).items()}
    raw_limits = torch.tensor(model.jnt_range[1:], dtype=torch.float32)
    center = raw_limits.mean(-1)
    half_width = (raw_limits[:, 1] - raw_limits[:, 0]) * 0.45
    limits = torch.stack((center - half_width, center + half_width), dim=-1)
    action, previous_action = torch.randn(2, samples, joints) * 0.2
    targets = {key: value.roll(1, 0) for key, value in bodies.items()}
    joint_state = rep.joint_state(state)
    reward = reward_module.tracking_reward(
        aux,
        bodies["body_pos_w"],
        matrix_from_quat(bodies["body_quat_w"]),
        bodies["body_lin_vel_w"],
        bodies["body_ang_vel_w"],
        joint_state,
        action,
        previous_action,
        targets["body_pos_w"],
        targets["body_quat_w"],
        targets["body_lin_vel_w"],
        targets["body_ang_vel_w"],
        limits,
        dt,
    )
    thresholds = SimpleNamespace(
        _anchor_index=0,
        _anchor_pos_threshold=env_cfg.terminations["anchor_pos"].params["threshold"],
        _anchor_ori_threshold=env_cfg.terminations["anchor_ori"].params["threshold"],
        _termination_softness=0.05,
    )
    termination = world_env.WorldModelEnv._compute_term_prob(
        thresholds,
        aux,
        targets["body_pos_w"],
        targets["body_quat_w"],
    )
    motion_view = SimpleNamespace(
        joint_pos=torch.from_numpy(motion["joint_pos"]),
        joint_vel=torch.from_numpy(motion["joint_vel"]),
        motion_ends=torch.from_numpy(motion["motion_ends"]),
        root_pos_w=torch.from_numpy(motion["body_pos_w"][:, 0]),
        root_quat_w=torch.from_numpy(motion["body_quat_w"][:, 0]),
    )
    times = torch.tensor([0, 46, 47, 48, 94, 95])
    goal_pos, goal_quat = representations.motion_anchor_window(motion_view, times)
    goal = representations._assemble_goal(
        representations.motion_goal_window(motion_view, times),
        full[:6, :3],
        full[:6, 3:7],
        goal_pos,
        goal_quat,
    )[0]
    tensors = {
        "full_state": full,
        "rep_state": state,
        "aux_state": aux,
        "prediction": prediction,
        "next_aux_state": next_aux,
        "next_rep_state": next_state,
        "reconstructed": rep.reconstruct_full_state(aux, state),
        "joint_state": joint_state,
        "default_joint_pos": default,
        "soft_joint_pos_limits": limits,
        "action": action,
        "last_action": previous_action,
        "reward": reward,
        "termination": termination,
        "goal_times": times,
        "goal": goal,
        "goal_indices": representations.goal_window_indices(motion_view, times),
        **bodies,
        **{f"target_{key}": value for key, value in targets.items()},
        **{f"loss_{key}": value for key, value in rep.loss_blocks(state, aux).items()},
    }
    learner_cfg = learner.WorldModelLearnerCfg()
    nominal = [getattr(learner_cfg, field) for field in rep.LOSS_WEIGHT_CONFIG.values()]
    body_joint = np.full(model.nbody - 1, -1, dtype=np.int64)
    body_joint[model.jnt_bodyid[1:] - 1] = np.arange(joints)
    np.savez_compressed(
        cfg.out / "reference.npz",
        **{key: value.detach().numpy() for key, value in tensors.items()},
        dt=np.float32(dt),
        bryson_weights=np.asarray([1 / value**2 for value in nominal], dtype=np.float32),
        anchor_pos_threshold=np.float32(thresholds._anchor_pos_threshold),
        anchor_ori_threshold=np.float32(thresholds._anchor_ori_threshold),
        robot_parents=model.body_parentid[1:] - 1,
        robot_body_joint=body_joint,
        robot_body_pos=model.body_pos[1:].astype(np.float32),
        robot_body_quat=model.body_quat[1:].astype(np.float32),
        robot_joint_axis=model.jnt_axis[1:].astype(np.float32),
        robot_joint_anchor=model.jnt_pos[1:].astype(np.float32),
        robot_joint_reference=model.qpos0[model.jnt_qposadr[1:]].astype(np.float32),
        robot_tracked_body_indices=np.asarray(tracked, dtype=np.int64),
    )
    files = [
        "representations.py",
        "tasks/env_cfgs.py",
        "utils/gencoord_helpers.py",
        "utils/helpers.py",
        "utils/tracking_reward.py",
        "envs/world_model_env.py",
        "learners/wm_learner.py",
    ]
    metadata = {
        "schema_version": 1,
        "seed": cfg.seed,
        "reference_commit": subprocess.check_output(
            ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
        ).strip(),
        "versions": {name: version(name) for name in ("torch", "mjlab", "mujoco", "numpy")},
        "sources_sha256": {
            name: hashlib.sha256((reference / "src/supermimic" / name).read_bytes()).hexdigest() for name in files
        },
        "fixture_sha256": {
            name: hashlib.sha256((cfg.out / name).read_bytes()).hexdigest() for name in ("motion.npz", "reference.npz")
        },
        "body_names": list(names),
        "loss_group_names": list(rep.LOSS_WEIGHT_CONFIG),
        "integration_substep_coeff": 1.0,
        "fk_oracle": "MuJoCo mj_forward + body 原点处的 mj_jac；发布版不含参考私有 fork 的 FK",
        "motion_note": "96 帧合成动作，仅用于 CPU 契约验收，不是 lafan_run1_subject5",
    }
    (cfg.out / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成夹具：{cfg.out.resolve()}")


if __name__ == "__main__":
    generate(tyro.cli(Config, description="从只读 SuperMimic 参考仓库重新生成对拍夹具。"))

"""用 PyTorch 检查点或纯 Actor ONNX 闭环评估单段外置动作。

每个环境只计从第 0 帧开始的首次尝试，失败后的重置不进入统计。
关闭探索噪声和初态扰动，保留三项域随机化与统一传感器噪声。
终止状态在手动重置前计分，不用短回合误差代替完整动作成功率。
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from torch import Tensor

from nanomimic.common.specs import Obs, SimConfig
from nanomimic.models.policy import Actor
from nanomimic.rollout.imagine import actor_goal
from nanomimic.rollout.sensor import HistoryRing, add_noise
from nanomimic.runtime.inference import OnnxActor
from nanomimic.runtime.paths import run_directory
from nanomimic.runtime.sim import Sim
from nanomimic.training.checkpoints import load_weights
from nanomimic.training.config import load_config


@dataclass(frozen=True)
class Config:
    # 检查点路径 <运行目录>/checkpoints/<迭代号>，与 --onnx 二选一。
    checkpoint: Path | None = None
    # 独立评估的并行尝试数，每行使用不同的域随机化与传感器噪声。
    num_envs: int = 64
    # 评估设备；CPU 可用于功能验证。
    device: str = "cuda:0"
    # 固定仿真和传感器噪声种子；对比检查点时保持相同。
    seed: int = 0
    # JSON 结果路径；默认放在运行目录的 evaluations/，不覆盖已有结果。
    out: Path | None = None
    # 纯 Actor ONNX 路径，在 CPU 上运行。
    onnx: Path | None = None
    # 单段参考动作，使用 ONNX 时须提供。
    motion_file: Path | None = None


class TrackingScore:
    """逐尝试累计误差及首次终止，失败后忽略该行的所有后续数据。"""

    def __init__(self, num_envs: int, names: tuple[str, ...], device: torch.device) -> None:
        self.names = names
        self.alive = torch.ones(num_envs, dtype=torch.bool, device=device)  # [E]
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.sums = torch.zeros(num_envs, len(names), dtype=torch.float64, device=device)  # [E, M]
        self.terminated = torch.zeros_like(self.alive)
        self.truncated = torch.zeros_like(self.alive)

    def add(self, errors: Tensor, terminated: Tensor, truncated: Tensor) -> None:
        if errors.shape != self.sums.shape or not torch.isfinite(errors[self.alive]).all():
            raise ValueError("跟踪误差形状错误或包含非有限值")
        # 包含失败的最后一步；无效行即便含 NaN 也不能污染有效统计。
        self.sums += torch.where(self.alive[:, None], errors.double(), 0)
        self.steps += self.alive
        self.terminated |= self.alive & terminated
        self.truncated |= self.alive & truncated
        self.alive &= ~(terminated | truncated)

    def summary(self, expected_steps: int, dt: float) -> dict[str, Any]:
        if expected_steps < 1 or bool((self.steps > expected_steps).any()) or bool((self.steps == 0).any()):
            raise ValueError("评估步数无效")
        if bool((self.alive & (self.steps != expected_steps)).any()):
            raise ValueError("尚未走完整段动作，不能统计成功率")

        def means(rows: Tensor) -> dict[str, float] | None:
            count = int(self.steps[rows].sum())
            return dict(zip(self.names, (self.sums[rows].sum(0) / count).tolist(), strict=True)) if count else None

        return {
            "trials": self.steps.numel(),
            "successes": int(self.alive.sum()),
            "success_rate": float(self.alive.float().mean()),
            "terminated_trials": int(self.terminated.sum()),
            "truncated_trials": int(self.truncated.sum()),
            "expected_steps": expected_steps,
            "mean_survival_seconds": float(self.steps.double().mean()) * dt,
            "mean_completed_fraction": float((self.steps.double() / expected_steps).mean()),
            "errors_all_attempted_steps": means(torch.ones_like(self.alive)),
            "errors_successful_trials": means(self.alive),
            "per_trial": [
                {"steps": steps, "success": success, "terminated": terminated, "truncated": truncated}
                for steps, success, terminated, truncated in zip(
                    self.steps.tolist(),
                    self.alive.tolist(),
                    self.terminated.tolist(),
                    self.truncated.tolist(),
                    strict=True,
                )
            ],
        }


def _reset_at_start(sim: Sim, rows: Tensor | None = None) -> Obs:
    """局部访问仿真边界以手动重置；发布版 reset 的命令计算会多推进一帧。"""
    sim._env.reset(env_ids=rows)
    sim._command.time_steps[slice(None) if rows is None else rows] = 0
    sim._command.update_relative_body_poses()
    return sim._snapshot(sim._env.observation_manager.compute(update_history=True))


@torch.no_grad()
def evaluate_policy(sim: Sim, actor: Actor | OnnxActor) -> dict[str, Any]:
    """从动作首帧跑到末帧，返回独立首次尝试的统计，不更新模型及归一化。"""
    if sim.motion.clip_ends.numel() != 1 or sim.motion.num_frames < 2:
        raise ValueError("完整动作评估需要至少两帧的单片段动作文件")
    command = sim._command
    command.cfg.sampling_mode = "start"
    command.cfg.pose_range = {}
    command.cfg.velocity_range = {}
    command.cfg.joint_position_range = (0.0, 0.0)
    sim._env.cfg.episode_length_s = (sim.motion.num_frames + 1) * sim.robot.dt
    sim._env.cfg.auto_reset = False
    obs = _reset_at_start(sim)
    if isinstance(actor, Actor):
        actor.eval()
    history = HistoryRing(actor.history_length)
    history.push(add_noise(obs.sensor))
    command._update_metrics()
    names = tuple(sorted(name for name in command.metrics if name.startswith("error_")))
    score = TrackingScore(sim.num_envs, names, sim.device)
    for frame in range(1, sim.motion.num_frames):
        action = actor(history.window(), actor_goal(obs.goal))
        obs, _, terminated, truncated = sim.step(action)
        if not bool(sim.motion_time[score.alive].eq(frame).all()):
            raise ValueError("评估中的动作时间发生跳转，不能计作完整跟踪")
        # step 已刷新真实 body 位姿；显式重算指标，避免读取命令管理器的上一帧缓存。
        command._update_metrics()
        errors = torch.stack([command.metrics[name] for name in names], dim=-1)
        score.add(errors, terminated, truncated)
        if not bool(score.alive.any()):
            break
        done = terminated | truncated
        if bool(done.any()):
            # 已失败行只为后续物理步清除手动重置要求，不再计作新尝试。
            obs = _reset_at_start(sim, done.nonzero().flatten())
        noisy = add_noise(obs.sensor)
        history.push(noisy)
        history.reset(done, noisy[done])
    return score.summary(sim.motion.num_frames - 1, sim.robot.dt)


def evaluate(cfg: Config) -> Path:
    """加载 Actor 与参考动作，写入可供对比的评估 JSON。"""
    if cfg.num_envs < 1:
        raise ValueError("评估尝试数必须大于零")
    if (cfg.checkpoint is None) == (cfg.onnx is None):
        raise ValueError("请在 --checkpoint 和 --onnx 中指定一个")
    checkpoint = cfg.checkpoint
    if checkpoint is None:
        assert cfg.onnx is not None
        if cfg.motion_file is None:
            raise ValueError("评估 ONNX 需要 --motion-file 指定外置动作")
        train_cfg = None
        model_file = cfg.onnx
        motion_file = cfg.motion_file
        run = model_file.parent.parent if model_file.parent.name == "exports" else model_file.parent
        label = f"{model_file.stem}-onnx-{motion_file.stem}"
    else:
        train_cfg = load_config(checkpoint)
        model_file = checkpoint / "actor.pt"
        motion_file = cfg.motion_file or train_cfg.motion_file
        if motion_file is None:
            raise ValueError("检查点缺少 motion_file")
        run = run_directory(checkpoint)
        label = checkpoint.name if cfg.motion_file is None else f"{checkpoint.name}-{motion_file.stem}"
    out = cfg.out or run / "evaluations" / f"eval-{label}-seed{cfg.seed}.json"
    if out.exists():
        raise FileExistsError(f"评估文件已存在，拒绝覆盖：{out}")
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    with Sim(SimConfig(motion_file, cfg.num_envs, cfg.device, cfg.seed)) as sim:
        if checkpoint is not None and train_cfg is not None:
            actor = Actor(len(sim.robot.joint_names), train_cfg.actor).to(sim.device)
            iteration = load_weights(checkpoint, "actor", actor, sim.device)
        else:
            actor = OnnxActor(model_file)
            if actor.joint_names != sim.robot.joint_names:
                raise ValueError("ONNX 关节顺序与 G1 资产不一致")
            iteration = actor.iteration
        # 初始化网络消耗 RNG；评估噪声从独立的固定种子开始。
        torch.manual_seed(cfg.seed)
        result = evaluate_policy(sim, actor)
        result.update(
            schema_version=1,
            checkpoint=str(checkpoint.resolve()) if checkpoint is not None else None,
            onnx=str(model_file.resolve()) if cfg.onnx is not None else None,
            iteration=iteration,
            motion_file=str(motion_file.resolve()),
            motion_sha256=hashlib.sha256(motion_file.read_bytes()).hexdigest(),
            actor_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),
            protocol={
                "seed": cfg.seed,
                "device": cfg.device,
                "actor_backend": "onnxruntime_cpu" if cfg.onnx is not None else "pytorch",
                "torch_version": torch.__version__,
                "sampling_mode": "start",
                "initial_state_perturbation": False,
                "domain_randomization": list(sim.ctx_spec.names),
                "sensor_noise": True,
                "exploration_noise": 0.0,
                "auto_reset": False,
                "body_names": list(sim.robot.body_names),
                "body_error": "重锚定后各跟踪 body 的欧氏距离均值，单位 m",
                "rotation_error": "各跟踪 body 旋转误差角均值，单位 rad",
                "joint_error": "全部关节误差向量的 L2 范数，位置 rad、速度 rad/s",
                "error_aggregation": "包含首次终止状态，按实际尝试的控制步加权；另报仅成功尝试的误差",
            },
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    errors = result["errors_all_attempted_steps"]
    print(
        f"评估迭代 {iteration}：完整动作成功 {result['successes']}/{result['trials']}，"
        f"平均存活 {result['mean_survival_seconds']:.3f} 秒，"
        f"尝试期间 body 位置误差 {errors['error_body_pos']:.6f} m；结果：{out}",
        flush=True,
    )
    return out


if __name__ == "__main__":
    evaluate(tyro.cli(Config, description="从第 0 帧评估完整单段动作的闭环成功率与跟踪误差。"))

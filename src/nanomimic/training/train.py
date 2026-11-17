"""训练排程、真实环境采集与标量输出。

每轮采集一步，按回放比例累积训练次数；每次先拟合世界模型再改进策略。
检查点不包含模拟器和回放，恢复后重新预热，参数、统计量和优化器保持连续。
每次启动使用独立时间戳目录，每轮记录指标，曲线与检查点同步更新。
"""

import json
from datetime import datetime
from itertools import count
from math import isfinite
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
from torch import nn

from nanomimic.common.specs import SimConfig
from nanomimic.models.model import Dynamics, represent
from nanomimic.models.policy import Actor, Critic
from nanomimic.rollout.buffer import ReplayBatch, ReplayBuffer
from nanomimic.rollout.imagine import actor_goal
from nanomimic.rollout.sensor import HistoryRing, add_noise
from nanomimic.runtime.paths import write_json
from nanomimic.runtime.sim import Sim
from nanomimic.training.checkpoints import load_checkpoint, save_checkpoint
from nanomimic.training.config import TrainConfig, config_dict, resolve_config
from nanomimic.training.curves import plot_training_curves
from nanomimic.training.learning import PolicyImprover, WorldModelFitter
from nanomimic.training.optim import Optimizers


class ScalarSink:
    """逐轮落盘指标、定期打印和绘图；W&B 仅在显式配置时发送标量。"""

    def __init__(self, project: str | None = None, *, output: Path | None = None) -> None:
        self._output = output
        self._history: list[dict[str, float]] = []
        self._plotted_iteration: int | None = None
        self._labels = {
            "wm_loss": "世界模型损失",
            "actor_loss": "策略损失",
            "critic_loss": "价值损失",
            "sim_reward": "仿真奖励",
            "done_fraction": "重置比例",
            "updates": "累计训练次数",
            "cuda_allocated_gb": "PyTorch 显存峰值 GB",
            "cuda_reserved_gb": "PyTorch 保留显存峰值 GB",
        }
        self._run: Any = None
        if project is not None:
            import wandb

            self._run = wandb.init(project=project)
        self._stream = None if output is None else (output / "metrics.jsonl").open("x", encoding="utf-8", buffering=1)

    def record(self, iteration: int, metrics: dict[str, float]) -> None:
        """记录每次采集迭代的实际指标，未执行训练时不补损失。"""
        if self._stream is not None:
            row = {"iteration": iteration, **metrics}
            self._stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self._history.append(row)

    def log(self, iteration: int, metrics: dict[str, float]) -> None:
        values = " | ".join(f"{self._labels.get(name, name)} {value:.6g}" for name, value in metrics.items())
        print(f"迭代 {iteration} | {values}", flush=True)
        if self._run is not None:
            self._run.log(metrics, step=iteration)

    def plot(self) -> None:
        """曲线覆盖更新到最近一轮；结束恰逢保存周期时不重复绘制。"""
        if self._output is not None and self._history:
            iteration = int(self._history[-1]["iteration"])
            if iteration != self._plotted_iteration:
                plot_training_curves(self._history, self._output, self._labels)
                self._plotted_iteration = iteration

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        if self._run is not None:
            self._run.finish()


def _create_run_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    for suffix in count():
        directory = root / (stamp if suffix == 0 else f"{stamp}_{suffix:02d}")
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return directory
    raise AssertionError("无法创建训练输出目录")


class Collector:
    def __init__(self, sim: Sim, dynamics: Dynamics, actor: Actor, buffer: ReplayBuffer, noise: float) -> None:
        self.sim, self.dynamics, self.actor, self.buffer = sim, dynamics, actor, buffer
        self.noise = noise
        self.obs = sim.reset()
        self.frame = add_noise(self.obs.sensor)
        self.history = HistoryRing(actor.history_length)
        self.history.push(self.frame)

    @torch.no_grad()
    def step(self) -> dict[str, float]:
        time, ctx = self.sim.motion_time, self.sim.read_ctx()
        goal = actor_goal(self.obs.goal)
        self.dynamics.state_norm.update(represent(self.obs.full_state))
        self.actor.obs_norm.update(self.frame)
        self.actor.goal_norm.update(goal)
        action = self.actor(self.history.window(), goal)
        action = action + self.noise * torch.randn_like(action)
        self.dynamics.action_norm.update(self.dynamics.process_action(action))
        next_obs, reward, terminated, truncated = self.sim.step(action)
        self.buffer.append(ReplayBatch(self.obs.full_state, self.frame, action, time, terminated, truncated, ctx))
        self.obs, self.frame = next_obs, add_noise(next_obs.sensor)
        self.history.push(self.frame)
        done = terminated | truncated
        self.history.reset(done, self.frame[done])
        return {"sim_reward": float(reward.mean()), "done_fraction": float(done.float().mean())}


class Trainer:
    def __init__(self, sim: Sim, cfg: TrainConfig) -> None:
        cfg.validate()
        self.cfg, self.sim = cfg, sim
        torch.manual_seed(cfg.seed)
        torch.set_float32_matmul_precision("high")
        joints, contexts = len(sim.robot.joint_names), sum(sim.ctx_spec.widths)
        self.dynamics = Dynamics(sim.robot, contexts, cfg.dynamics).to(sim.device)
        self.actor = Actor(joints, cfg.actor).to(sim.device)
        self.critic = Critic(joints, contexts, cfg.critic).to(sim.device)
        self.buffer = ReplayBuffer(sim.num_envs, cfg.buffer_capacity)
        self.fitter = WorldModelFitter(self.dynamics, cfg)
        self.improver = PolicyImprover(self.dynamics, self.actor, self.critic, sim.motion, sim.robot, cfg)
        self.collector = Collector(sim, self.dynamics, self.actor, self.buffer, cfg.policy.exploration_noise)
        self.iteration, self.updates, self.credit = 0, 0, 0.0
        self.last_checkpoint = cfg.resume
        self.last_saved_iteration = -1
        if cfg.resume is not None:
            progress = load_checkpoint(cfg.resume, self.models(), sim.device)
            self.iteration, self.updates = int(progress["iteration"]), int(progress["updates"])
            self.credit = float(progress["credit"])
            self.last_saved_iteration = self.iteration
            if cfg.max_iterations < self.iteration:
                raise ValueError("max_iterations 不能小于检查点中已完成的迭代号")
        # 配置保留输出根目录，续训时沿用根目录创建新的运行，避免嵌套时间戳。
        self.output = _create_run_directory(cfg.output)
        for name in ("checkpoints", "evaluations", "exports"):
            (self.output / name).mkdir()
        write_json(self.output / "config.json", config_dict(cfg))
        write_json(
            self.output / "run.json",
            {
                "schema_version": 1,
                "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "resume": None if cfg.resume is None else str(cfg.resume),
            },
        )
        if sim.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(sim.device)

    def models(self) -> dict[str, tuple[nn.Module, Optimizers]]:
        return {
            "world_model": (self.dynamics, self.fitter.optim),
            "actor": (self.actor, self.improver.actor_optim),
            "critic": (self.critic, self.improver.critic_optim),
        }

    def step(self) -> dict[str, float]:
        metrics = self.collector.step()
        self.iteration += 1
        wm = self.cfg.wm
        self.credit += self.cfg.replay_ratio * self.sim.num_envs / (wm.num_batches * wm.batch * wm.horizon)
        if self.buffer.can_sample(wm.horizon + 1):
            rounds = int(self.credit + 1e-12)
            totals: dict[str, float] = {}
            for _ in range(rounds):
                current = self.fitter.fit(self.buffer) | self.improver.improve(self.buffer)
                for name, value in current.items():
                    totals[name] = totals.get(name, 0.0) + value
                self.updates += 1
                self.credit = max(0.0, self.credit - 1)
            if rounds:
                metrics.update({name: value / rounds for name, value in totals.items()})
        metrics["updates"] = float(self.updates)
        if self.sim.device.type == "cuda":
            metrics["cuda_allocated_gb"] = torch.cuda.max_memory_allocated(self.sim.device) / 1e9
            metrics["cuda_reserved_gb"] = torch.cuda.max_memory_reserved(self.sim.device) / 1e9
        if not all(isfinite(value) for value in metrics.values()):
            raise FloatingPointError("采集或训练标量出现非有限值")
        return metrics

    def save(self) -> Path:
        if self.last_saved_iteration != self.iteration:
            directory = self.output / "checkpoints" / f"{self.iteration:08d}"
            progress = {"iteration": self.iteration, "updates": self.updates, "credit": self.credit}
            save_checkpoint(directory, self.cfg, self.models(), progress)
            self.last_checkpoint, self.last_saved_iteration = directory, self.iteration
        assert self.last_checkpoint is not None
        return self.last_checkpoint

    def run(self, sink: ScalarSink) -> Path:
        while self.iteration < self.cfg.max_iterations:
            metrics = self.step()
            sink.record(self.iteration, metrics)
            if self.iteration % self.cfg.log_interval == 0:
                sink.log(self.iteration, metrics)
            if self.iteration % self.cfg.save_interval == 0:
                self.save()
                sink.plot()
        result = self.save()
        sink.plot()
        return result


def train(cfg: TrainConfig) -> Path:
    cfg = resolve_config(cfg)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用；CPU 验收请显式设置 --device cpu 并缩小环境数、批量与网络")
    assert cfg.motion_file is not None
    with Sim(SimConfig(cfg.motion_file, cfg.num_envs, cfg.device, cfg.seed)) as sim:
        trainer = Trainer(sim, cfg)
        print(f"训练输出目录：{trainer.output}", flush=True)
        sink = ScalarSink(cfg.wandb_project, output=trainer.output)
        try:
            result = trainer.run(sink)
        finally:
            sink.close()
    print(f"训练已结束，检查点：{result}", flush=True)
    return result

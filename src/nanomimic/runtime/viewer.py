"""真实仿真回放与 env 0 的世界模型闭环虚影。

真实与预测两侧独立维护带噪历史，预测动作只读取预测状态和目标。
回合结束、手动重置或达到 rollout 周期后重新对齐，不更新网络或归一化。
"""

from copy import deepcopy
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from mjlab.viewer import BaseViewer, EnvProtocol, NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.debug_visualizer import DebugVisualizer
from torch import Tensor

from nanomimic.common.specs import Obs, SimConfig
from nanomimic.models.model import Dynamics, aux_state, reconstruct_full_state, represent
from nanomimic.models.policy import Actor
from nanomimic.rollout.imagine import actor_goal, wm_goal
from nanomimic.rollout.sensor import HistoryRing, add_noise, synth_proprio
from nanomimic.runtime.sim import Sim
from nanomimic.training.checkpoints import load_weights
from nanomimic.training.config import load_config


class GhostRollout:
    """只预测 env 0；真实侧仍支持并行环境，其他行的 done 不重置虚影。"""

    def __init__(self, sim: Sim, actor: Actor, dynamics: Dynamics, num_steps_rollout: int = 1_000_000) -> None:
        if num_steps_rollout < 1:
            raise ValueError("num_steps_rollout 必须大于零")
        self.sim, self.actor, self.dynamics = sim, actor.eval(), dynamics.eval()
        self.num_steps_rollout = num_steps_rollout
        self.real_history, self.history = HistoryRing(actor.history_length), HistoryRing(actor.history_length)
        self.steps = self.total_steps = self.alignments = 0
        self.done_alignments = self.periodic_alignments = 0
        self.model = deepcopy(sim._env.sim.mj_model)
        self.model.geom_rgba[:] = [0.5, 0.5, 0.8, 0.8]
        indexing = sim._env.scene["robot"].indexing
        self.root_indices = indexing.free_joint_q_adr.cpu().numpy()
        self.joint_indices = indexing.joint_q_adr.cpu().numpy()
        self.qpos = self.model.qpos0.copy()
        self._original_visualizer = sim._env.update_visualizers
        self.obs = sim.reset()
        self.reset_from_sim()
        sim._env.update_visualizers = self.draw

    def _align(self, obs: Obs) -> None:
        self.state, self.aux = represent(obs.full_state[:1]), aux_state(obs.full_state[:1]).clone()
        self.time, self.ctx = self.sim.motion_time[:1], self.sim.read_ctx()[:1]
        self.last_action = obs.sensor[:1, -len(self.sim.robot.joint_names) :].clone()
        frame = add_noise(synth_proprio(self.state, self.aux, self.last_action, self.sim.robot.default_joint_pos))
        self.history.push(frame)
        self.history.reset(torch.tensor([0], device=self.sim.device), frame)
        self.steps = 0
        self.alignments += 1
        self._update_qpos()

    @torch.no_grad()
    def reset_from_sim(self) -> None:
        """手动重置后刷新两侧历史；不再次重置仿真或改变抽到的动作起点。"""
        self.obs = self.sim._snapshot(self.sim._env.get_observations())
        frame = add_noise(self.obs.sensor)
        self.real_history.push(frame)
        rows = torch.arange(self.sim.num_envs, device=self.sim.device)
        self.real_history.reset(rows, frame)
        self._align(self.obs)
        self._episode_step = int(self.sim._env.episode_length_buf[0])

    def _update_qpos(self) -> None:
        # 根与关节按发布版实体的 qpos 地址写入，不假设 MuJoCo 的关节存储顺序。
        full = reconstruct_full_state(self.aux, self.state)[0].cpu().numpy()
        self.qpos[self.root_indices] = full[:7]
        self.qpos[self.root_indices[:3]] += self.sim._env.scene.env_origins[0].cpu().numpy()
        self.qpos[self.joint_indices] = full[7 : 7 + len(self.joint_indices)]

    @torch.no_grad()
    def step(self) -> Tensor:
        """先真实推进，再独立预测一步；done 时直接对齐返回的新回合观测。"""
        # Viser 的动作控件可以绕过 viewer.reset_environment，直接重置底层环境。
        if int(self.sim._env.episode_length_buf[0]) != self._episode_step:
            self.reset_from_sim()
        action = self.actor(self.real_history.window(), actor_goal(self.obs.goal))
        obs, _, terminated, truncated = self.sim.step(action)
        goal = actor_goal(wm_goal(self.sim.motion, self.aux, self.time))
        prediction_action = self.actor(self.history.window(), goal)
        self.aux, self.state = self.dynamics(self.state, prediction_action, self.aux, self.ctx)
        self.last_action = prediction_action
        self.time = torch.minimum(self.time + 1, self.sim.motion.last_frame(self.time))
        self.steps += 1
        self.total_steps += 1
        frame = add_noise(obs.sensor)
        self.real_history.push(frame)
        done = terminated | truncated
        self.real_history.reset(done, frame[done])
        if bool(done[0]) or self.steps >= self.num_steps_rollout:
            self.done_alignments += int(bool(done[0]))
            self.periodic_alignments += int(not bool(done[0]))
            self._align(obs)
        else:
            sensor = synth_proprio(self.state, self.aux, prediction_action, self.sim.robot.default_joint_pos)
            self.history.push(add_noise(sensor))
            self._update_qpos()
        self.obs = obs
        self._episode_step = int(self.sim._env.episode_length_buf[0])
        if not np.isfinite(self.qpos).all():
            raise ValueError("世界模型虚影出现非有限状态")
        return done

    def draw(self, visualizer: DebugVisualizer) -> None:
        self._original_visualizer(visualizer)
        if 0 in visualizer.get_env_indices(self.sim.num_envs):
            visualizer.add_ghost_mesh(self.qpos, model=self.model, alpha=0.8, label="wm_prediction")

    def close(self) -> None:
        self.sim._env.update_visualizers = self._original_visualizer


def _advance_viewer(viewer: BaseViewer, rollout: GhostRollout) -> bool:
    """沿用 mjlab 的暂停/单步调度，失败时暂停并在终端明确说明。"""
    try:
        rollout.step()
        viewer._step_count += 1
        viewer._stats_steps += 1
        return True
    except Exception as error:
        viewer._last_error = f"回放失败：{type(error).__name__}: {error}"
        print(viewer._last_error, flush=True)
        viewer.pause()
        return False


class NativeGhostViewer(NativeMujocoViewer):
    def __init__(self, rollout: GhostRollout) -> None:
        # 发布版 EnvProtocol 把只读属性标为可变成员；运行时接受原始环境。
        super().__init__(cast(EnvProtocol, rollout.sim._env), rollout.actor)
        self.rollout = rollout

    def _execute_step(self) -> bool:
        return _advance_viewer(self, self.rollout)

    def reset_environment(self) -> None:
        super().reset_environment()
        self.rollout.reset_from_sim()


class ViserGhostViewer(ViserPlayViewer):
    def __init__(self, rollout: GhostRollout) -> None:
        super().__init__(cast(EnvProtocol, rollout.sim._env), rollout.actor)
        self.rollout = rollout

    def _execute_step(self) -> bool:
        return _advance_viewer(self, self.rollout)

    def reset_environment(self) -> None:
        super().reset_environment()
        self.rollout.reset_from_sim()


def play(
    checkpoint: Path,
    viewer: Literal["native", "viser"] = "native",
    device: str | None = None,
    num_steps_rollout: int = 1_000_000,
    num_steps: int | None = None,
) -> None:
    """加载 Actor 与世界模型供单环境回放；num_steps 用于有限步数的实际查看器验收。"""
    if num_steps_rollout < 1 or (num_steps is not None and num_steps < 1):
        raise ValueError("回放与对齐步数必须大于零")
    cfg = load_config(checkpoint)
    if cfg.motion_file is None:
        raise ValueError("检查点缺少 motion_file")
    target = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    with Sim(SimConfig(cfg.motion_file, num_envs=1, device=target, seed=cfg.seed)) as sim:
        actor = Actor(len(sim.robot.joint_names), cfg.actor).to(sim.device)
        dynamics = Dynamics(sim.robot, sum(sim.ctx_spec.widths), cfg.dynamics).to(sim.device)
        iteration = load_weights(checkpoint, "actor", actor, sim.device)
        if load_weights(checkpoint, "world_model", dynamics, sim.device) != iteration:
            raise ValueError("Actor 与世界模型检查点的迭代号不一致")
        rollout = GhostRollout(sim, actor, dynamics, num_steps_rollout)
        try:
            window = NativeGhostViewer(rollout) if viewer == "native" else ViserGhostViewer(rollout)
            print(f"回放迭代 {iteration}：设备 {sim.device}，{viewer} 查看器，虚影只预测 env 0。", flush=True)
            window.run(num_steps=num_steps)
            if window._last_error is not None:
                raise ValueError(window._last_error)
            print(
                f"回放结束：{rollout.total_steps} 步，回合对齐 {rollout.done_alignments} 次，"
                f"定期对齐 {rollout.periodic_alignments} 次。"
            )
        finally:
            rollout.close()

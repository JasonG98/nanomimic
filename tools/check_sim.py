"""T2 随机动作验收入口，可在 CPU 或 CUDA 上运行。

检查所有公开张量的形状、有限性和随机化范围。
输出当前设备与步数，不以合成动作结果代表训练效果。
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
import tyro

from nanomimic.common.specs import SimConfig
from nanomimic.runtime.sim import Sim


@dataclass(frozen=True)
class Config(SimConfig):
    """小规模 CPU 验收默认配置；GPU 验收时显式覆盖设备和环境数。"""

    motion_file: Path = Path("tests/fixtures/motion.npz")
    num_envs: int = 2
    device: str = "cpu"
    steps: int = 100


def check(cfg: Config) -> None:
    """运行随机动作并检查仿真边界契约。"""
    if cfg.steps < 1:
        raise ValueError("steps 必须大于零")
    with Sim(cfg) as sim:
        sim.reset()
        generator = torch.Generator(device=sim.device).manual_seed(cfg.seed)
        done_count = 0
        if sim.device.type == "cuda":
            torch.cuda.synchronize(sim.device)
        start = perf_counter()
        for _ in range(cfg.steps):
            action = torch.rand(sim.num_envs, 29, device=sim.device, generator=generator) * 2 - 1
            obs, reward, terminated, truncated = sim.step(action)
            assert obs.full_state.shape == (sim.num_envs, 71)
            assert obs.sensor.shape == (sim.num_envs, 93)
            assert obs.goal.shape == (sim.num_envs, 670)
            assert reward.shape == terminated.shape == truncated.shape == (sim.num_envs,)
            for value in (obs.full_state, obs.sensor, obs.goal, reward):
                assert torch.isfinite(value).all(), "仿真输出出现非有限值"
            context = sim.read_ctx()
            assert context.shape == (sim.num_envs, 33)
            assert context.abs().max() <= 1 + 1e-6, "随机化上下文超出归一化范围"
            done_count += int((terminated | truncated).sum())
        if sim.device.type == "cuda":
            torch.cuda.synchronize(sim.device)
        elapsed = perf_counter() - start
        print(f"验收通过：设备 {sim.device}，{sim.num_envs} 个环境，{cfg.steps} 步，重置 {done_count} 次。")
        print(f"含契约检查耗时 {elapsed:.2f} 秒，环境步数/秒 {cfg.steps * sim.num_envs / elapsed:.1f}。")


if __name__ == "__main__":
    check(tyro.cli(Config, description="在 CPU 或 CUDA 上运行 G1 随机动作契约验收。"))

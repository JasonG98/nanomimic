"""训练标量输出，终端默认启用，W&B 需显式配置。

仅记录标量，不上传模型、动作库或其他产物。
W&B 延迟导入，未指定项目时不建立网络会话。
"""

from typing import Any


class ScalarSink:
    def __init__(self, project: str | None = None) -> None:
        self._run: Any = None
        if project is not None:
            import wandb

            self._run = wandb.init(project=project)

    def log(self, iteration: int, metrics: dict[str, float]) -> None:
        labels = {
            "wm_loss": "世界模型损失",
            "actor_loss": "策略损失",
            "critic_loss": "价值损失",
            "sim_reward": "仿真奖励",
            "done_fraction": "重置比例",
            "updates": "累计训练次数",
            "cuda_allocated_gb": "PyTorch 显存峰值 GB",
            "cuda_reserved_gb": "PyTorch 保留显存峰值 GB",
        }
        values = " | ".join(f"{labels.get(name, name)} {value:.6g}" for name, value in metrics.items())
        print(f"迭代 {iteration} | {values}", flush=True)
        if self._run is not None:
            self._run.log(metrics, step=iteration)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()

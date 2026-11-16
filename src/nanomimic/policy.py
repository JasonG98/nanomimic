"""单任务 Actor 与特权 Critic。

Actor 只读取帧主序 proprio 与不含相对位置的目标，潜变量限制在 [-1, 1]。
观测及目标归一化显式更新；Critic 输入额外包含模型状态与随机化上下文。
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nanomimic.networks import RunningNorm, mlp


@dataclass(frozen=True)
class ActorConfig:
    """历史默认值取参考 BaseRunnerCfg.history_length。"""

    history_length: int = 1
    hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = 32


@dataclass(frozen=True)
class CriticConfig:
    hidden: tuple[int, ...] = (256, 128)


class Actor(nn.Module):
    def __init__(self, joints: int, cfg: ActorConfig = ActorConfig()) -> None:
        super().__init__()
        if cfg.history_length < 1:
            raise ValueError("历史长度必须大于零")
        self.history_length = cfg.history_length
        self.obs_norm = RunningNorm(6 + 3 * joints)
        self.goal_norm = RunningNorm(10 * (2 * joints + 6))
        self.encoder = mlp(10 * (2 * joints + 6), cfg.hidden, cfg.latent_dim)
        self.decoder = mlp(cfg.latent_dim + cfg.history_length * (6 + 3 * joints), cfg.hidden, joints)

    def forward(self, history: Tensor, goal: Tensor) -> Tensor:
        latent = self.encoder(self.goal_norm(goal)).tanh()
        return self.decoder(torch.cat((latent, self.obs_norm(history).flatten(1)), dim=-1))


class Critic(nn.Module):
    def __init__(self, joints: int, ctx_dim: int, cfg: CriticConfig = CriticConfig()) -> None:
        super().__init__()
        self.mlp = mlp(10 + 2 * joints + 10 * (2 * joints + 6) + ctx_dim, cfg.hidden, 1)

    @staticmethod
    def observation(state: Tensor, actor_goal: Tensor, ctx: Tensor) -> Tensor:
        return torch.cat((state, actor_goal, ctx), dim=-1)

    def forward(self, observation: Tensor) -> Tensor:
        return self.mlp(observation).squeeze(-1)

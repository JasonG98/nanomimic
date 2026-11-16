"""小型网络公共构件与显式更新的在线归一化。

前向计算不修改统计量；只有采集阶段调用 update。
均值、总体方差与样本数均为 buffer，随 state_dict 保存及迁移设备。
"""

import torch
from torch import Tensor, nn


class RunningNorm(nn.Module):
    """合并批次矩的在线归一化，标准差下限为 0.01。"""

    mean: Tensor
    var: Tensor
    count: Tensor

    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, data: Tensor) -> None:
        data = data.detach().reshape(-1, self.mean.numel())
        if data.shape[0] == 0:
            return
        variance, mean = torch.var_mean(data, dim=0, correction=0)
        total = self.count + data.shape[0]
        ratio = data.shape[0] / total
        delta = mean - self.mean
        self.var.copy_((1 - ratio) * self.var + ratio * variance + ratio * (1 - ratio) * delta.square())
        self.mean.add_(ratio * delta)
        self.count.copy_(total)

    def forward(self, data: Tensor) -> Tensor:
        return (data - self.mean) / self.var.clamp_min(1e-4).sqrt()


def mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
    """隐藏层统一使用 Linear、LayerNorm、GELU，末层不加激活。"""
    layers: list[nn.Module] = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)

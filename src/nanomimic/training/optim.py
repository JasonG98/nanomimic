"""每个 MLP 的 Muon / Adam 参数分组和受控优化步骤。

仅中间 Linear 权重进入 Muon，首末层、偏置及 LayerNorm 使用 Adam。
梯度先裁剪范数，再按需裁剪元素；非有限损失或梯度直接报错。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn


class Optimizers:
    def __init__(self, module: nn.Module, lr: float, beta2: float = 0.999) -> None:
        self.parameters = list(module.parameters())
        hidden = []
        for child in module.modules():
            if isinstance(child, nn.Sequential):
                linear = [layer for layer in child.children() if isinstance(layer, nn.Linear)]
                hidden.extend(layer.weight for layer in linear[1:-1])
        selected = {id(parameter) for parameter in hidden}
        other = [parameter for parameter in self.parameters if id(parameter) not in selected]
        self.optimizers: list[torch.optim.Optimizer] = [torch.optim.Adam(other, lr=lr, betas=(0.9, beta2))]
        if hidden:
            self.optimizers.append(torch.optim.Muon(hidden, lr=lr, weight_decay=0, adjust_lr_fn="match_rms_adamw"))

    def step(self, loss: Tensor, element_clip: float | None = None) -> float:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("训练损失出现非有限值")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True)
        if element_clip is not None:
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.clamp_(-element_clip, element_clip)
        for optimizer in self.optimizers:
            optimizer.step()
        return float(norm)

    def state_dict(self) -> list[dict[str, Any]]:
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, states: list[dict[str, Any]]) -> None:
        for optimizer, state in zip(self.optimizers, states, strict=True):
            # 恢复动量等状态；本次配置显式覆盖的学习率等超参仍应生效。
            choices = [
                {key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups
            ]
            optimizer.load_state_dict(state)
            for group, settings in zip(optimizer.param_groups, choices, strict=True):
                group.update(settings)


@contextmanager
def frozen_parameters(*modules: nn.Module) -> Iterator[None]:
    """冻结参数更新但保留网络对输入的导数，退出后恢复原标志。"""
    parameters = [parameter for module in modules for parameter in module.parameters()]
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, previous, strict=True):
            parameter.requires_grad_(enabled)

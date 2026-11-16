"""按环境存储的环形回放缓冲与向量化采样。

槽 t 保存观测 s_t、已加噪 sensor_t、发出的 a_t 与该动作产生的 done_t。
done_t 后的槽属于新回合；采样不把自动重置观测当作旧回合后继。
"""

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass(frozen=True)
class ReplayBatch:
    """append 时前缀为 [E]；窗口采样后前缀为 [B, L]。"""

    full_state: Tensor
    sensor: Tensor
    action: Tensor
    motion_time: Tensor
    terminated: Tensor
    truncated: Tensor
    ctx: Tensor


@dataclass(frozen=True)
class StartBatch:
    """起点及含当前帧的历史；mask 的 True 表示真实历史、False 表示填充。"""

    full_state: Tensor  # [B, 13+2J]
    motion_time: Tensor  # [B]
    ctx: Tensor  # [B, C]
    sensor: Tensor  # [B, H, D]，上一动作已包含在 sensor 中
    action: Tensor  # [B, H, J]，对应各历史槽发出的动作
    mask: Tensor  # [B, H]


class ReplayBuffer:
    """延迟分配固定容量，所有环境每次同时写入一个时间槽。"""

    def __init__(self, num_envs: int, capacity: int = 1000) -> None:
        if num_envs < 1 or capacity < 1:
            raise ValueError("环境数与缓冲容量必须大于零")
        self.num_envs, self.capacity = num_envs, capacity
        self.size, self.cursor = 0, 0
        self._data: dict[str, Tensor] = {}

    @torch.no_grad()
    def append(self, batch: ReplayBatch) -> None:
        values = {field.name: getattr(batch, field.name) for field in fields(batch)}
        if any(value.shape[0] != self.num_envs for value in values.values()):
            raise ValueError("写入批次必须包含全部环境")
        if not self._data:
            self._data = {
                name: value.new_empty((self.num_envs, self.capacity, *value.shape[1:]))
                for name, value in values.items()
            }
        for name, value in values.items():
            self._data[name][:, self.cursor].copy_(value.detach())
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _physical(self, logical: Tensor) -> Tensor:
        return (logical + self.cursor - self.size) % self.capacity

    def _done(self) -> Tensor:
        if self.size == 0:
            raise ValueError("缓冲为空，尚不能采样")
        data = self._data["terminated"] | self._data["truncated"]
        times = torch.arange(self.size, device=data.device)
        return data[:, self._physical(times)]

    def _valid_windows(self, length: int) -> Tensor:
        if length < 1 or length > self.size:
            raise ValueError("窗口长度必须在 1 到当前缓冲长度之间")
        done = self._done()
        prefix = torch.cat((done.new_zeros((self.num_envs, 1), dtype=torch.long), done.cumsum(1)), dim=1)
        starts = torch.arange(self.size - length + 1, device=done.device)
        # 仅内部 L-1 条边不能 done；最后一槽的动作并非窗口内的边。
        return prefix[:, starts + length - 1] == prefix[:, starts]

    def can_sample(self, length: int) -> bool:
        return 1 <= length <= self.size and bool(self._valid_windows(length).any())

    def _choose(self, valid: Tensor, batch: int) -> tuple[Tensor, Tensor]:
        """先均衡覆盖可用环境，再在每个环境的合法位置均匀抽样。"""
        if batch < 1:
            raise ValueError("采样批量必须大于零")
        counts = valid.sum(1)
        available = (counts > 0).nonzero().flatten()
        if available.numel() == 0:
            raise ValueError("当前没有不跨回合的有效窗口")
        available = available[torch.randperm(available.numel(), device=valid.device)]
        envs = available[torch.arange(batch, device=valid.device) % available.numel()]
        ranks = (torch.rand(batch, device=valid.device) * counts[envs]).long() + 1
        times = torch.searchsorted(valid[envs].cumsum(1), ranks[:, None]).squeeze(1)
        return envs, times

    def sample_window(self, length: int, batch: int) -> ReplayBatch:
        """返回 [B, L, ...]；用 done 累加和过滤，支持环形覆盖。"""
        envs, times = self._choose(self._valid_windows(length), batch)
        slots = self._physical(times[:, None] + torch.arange(length, device=times.device))
        return ReplayBatch(**{name: value[envs[:, None], slots] for name, value in self._data.items()})

    def sample_start(self, batch: int, history: int) -> StartBatch:
        if history < 1:
            raise ValueError("历史长度必须大于零")
        done = self._done()
        envs, times = self._choose(torch.ones_like(done), batch)
        indices = torch.arange(self.size, device=done.device)
        ends = torch.where(done, indices, -1).cummax(1).values
        # 起点自身的 done 属于尚未发生的动作，只看它之前的边。
        beginnings = torch.cat((ends.new_zeros((self.num_envs, 1)), ends[:, :-1] + 1), dim=1)
        first = beginnings[envs, times]
        requested = times[:, None] + torch.arange(1 - history, 1, device=done.device)
        mask = requested >= first[:, None]
        slots = self._physical(torch.maximum(requested, first[:, None]))
        current = self._physical(times)
        return StartBatch(
            self._data["full_state"][envs, current],
            self._data["motion_time"][envs, current],
            self._data["ctx"][envs, current],
            self._data["sensor"][envs[:, None], slots],
            self._data["action"][envs[:, None], slots],
            mask,
        )

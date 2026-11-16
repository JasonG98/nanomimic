"""T4 历史、噪声与回放边界验收。

使用携带环境和时间标识的状态，检查跨回合、覆盖和动作对齐。
统计采样规模仅用于 CPU 单测，不修改默认训练容量与历史配置。
"""

import pytest
import torch
from torch import Tensor

from nanomimic.buffer import ReplayBatch, ReplayBuffer
from nanomimic.sensor import HistoryRing, add_noise, noise_half_width, synth_proprio
from nanomimic.specs import RobotSpec


def test_synth_and_noise(reference: dict[str, Tensor], robot: RobotSpec) -> None:
    state, aux, action = reference["rep_state"], reference["aux_state"], reference["last_action"]
    frame = synth_proprio(state, aux, action, robot.default_joint_pos)
    expected = torch.cat(
        (state[:, 1:4], state[:, 7:10], state[:, 10:39] - robot.default_joint_pos, state[:, 39:], action), dim=-1
    )
    torch.testing.assert_close(frame, expected, atol=1e-5, rtol=0)
    noisy = add_noise(frame)
    assert ((noisy - frame).abs() <= noise_half_width(frame) + 1e-7).all()
    assert torch.equal(noisy[:, -29:], action)
    assert not torch.equal(noisy[:, :6], frame[:, :6])


def test_history_reset_and_autograd() -> None:
    history = HistoryRing(3)
    seed = torch.randn(4, 9, requires_grad=True)
    history.push(seed)
    torch.testing.assert_close(history.window(), seed[:, None].expand(-1, 3, -1))
    history.push(seed + 1)
    history.push(seed + 2)
    snapshot = history.window()
    reset = add_noise(torch.zeros(2, 9))
    history.reset(torch.tensor([1, 3]), reset)
    for row in (0, 2):
        torch.testing.assert_close(history.window()[row], torch.stack((seed[row], seed[row] + 1, seed[row] + 2)))
    torch.testing.assert_close(history.window()[[1, 3]], reset[:, None].expand(-1, 3, -1))
    assert not torch.equal(snapshot[1], history.window()[1])
    history.window().sum().backward()
    assert seed.grad is not None and torch.isfinite(seed.grad).all()


def _fill(capacity: int, steps: int) -> ReplayBuffer:
    buffer = ReplayBuffer(3, capacity)
    for time in range(steps):
        ids = torch.arange(3) * 100 + time
        value = ids.float()[:, None]
        # 环境 0 每 3 步终止，环境 1 每 4 步截断，环境 2 连续。
        buffer.append(
            ReplayBatch(
                value,
                value + 0.1,
                value + 0.2,
                ids,
                torch.tensor([time % 3 == 2, False, False]),
                torch.tensor([False, time % 4 == 3, False]),
                value + 0.3,
            )
        )
    return buffer


@pytest.mark.parametrize("capacity,steps", [(12, 10), (7, 30)])
def test_windows_do_not_cross_done(capacity: int, steps: int) -> None:
    buffer = _fill(capacity, steps)
    windows = buffer.sample_window(3, 120)
    assert not (windows.terminated[:, :-1] | windows.truncated[:, :-1]).any()
    assert (windows.motion_time[:, 1:] - windows.motion_time[:, :-1] == 1).all()
    envs = windows.motion_time[:, 0] // 100
    assert torch.equal(envs.bincount(), torch.tensor([40, 40, 40]))
    assert (windows.motion_time % 100 >= max(0, steps - capacity)).all()
    torch.testing.assert_close(windows.action, windows.full_state + 0.2)


@pytest.mark.parametrize("capacity,steps", [(12, 10), (7, 30)])
def test_start_padding_and_mask(capacity: int, steps: int) -> None:
    buffer = _fill(capacity, steps)
    starts = buffer.sample_start(600, 5)
    for row in range(600):
        ident = int(starts.motion_time[row])
        env, time = divmod(ident, 100)
        period = {0: 3, 1: 4}.get(env)
        beginning = 0 if period is None else time // period * period
        first = max(beginning, steps - capacity, 0)
        requested = torch.arange(time - 4, time + 1)
        expected = requested.clamp_min(first) + env * 100
        torch.testing.assert_close(starts.sensor[row, :, 0], expected.float() + 0.1)
        torch.testing.assert_close(starts.action[row, :, 0], expected.float() + 0.2)
        assert torch.equal(starts.mask[row], requested >= first)
    assert starts.mask[:, -1].all()


def test_empty_invalid_and_unavailable_environments() -> None:
    buffer = ReplayBuffer(2, 4)
    assert not buffer.can_sample(2)
    with pytest.raises(ValueError, match="缓冲为空"):
        buffer.sample_start(1, 1)
    for _ in range(4):
        frame = torch.tensor([[0.0], [1.0]])
        buffer.append(
            ReplayBatch(
                frame,
                frame,
                frame,
                torch.zeros(2, dtype=torch.long),
                torch.tensor([True, False]),
                torch.zeros(2, dtype=torch.bool),
                frame,
            )
        )
    windows = buffer.sample_window(4, 10)
    assert (windows.full_state == 1).all()
    assert buffer.sample_window(1, 2).full_state.shape == (2, 1, 1)
    assert buffer.sample_start(2, 1).mask.all()
    with pytest.raises(ValueError, match="窗口长度"):
        buffer.sample_window(5, 1)

"""训练配置、JSON 契约与恢复时的配置合并。

所有默认训练规模保持 SPEC；CPU 验收通过 CLI 显式覆盖。
schema_version 固定为 1，路径写成字符串，网络宽度从 JSON 恢复为元组。
"""

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from nanomimic.models.model import DynamicsConfig
from nanomimic.models.policy import ActorConfig, CriticConfig


@dataclass(frozen=True)
class WorldModelConfig:
    # 世界模型学习率。
    lr: float = 2e-4
    # 每个监督批次的窗口数。
    batch: int = 2048
    # 自回归预测步数，缓冲实际取 horizon+1 帧。
    horizon: int = 8
    # 每次训练中的世界模型批次数。
    num_batches: int = 10


@dataclass(frozen=True)
class PolicyTrainConfig:
    # Actor 与 Critic 的学习率。
    lr: float = 2e-4
    # 每次想象的起点数。
    batch: int = 2048
    # 每个起点的想象步数。
    horizon: int = 32
    # 采集与想象共用的动作高斯噪声标准差。
    exploration_noise: float = 0.1
    # 回报折扣，必须小于 1。
    gamma: float = 0.99
    # GAE 的 lambda。
    lam: float = 0.95
    # 策略侧 Adam 的二阶矩衰减。
    beta2: float = 0.95
    # 范数裁剪后逐元素梯度裁剪半宽。
    grad_clip: float = 0.1
    # 每次改进中的价值网络训练轮数。
    critic_epochs: int = 5
    # 每轮价值网络的 minibatch 数。
    critic_minibatches: int = 4
    # 对每个想象步启用激活重算以降低显存。
    checkpoint: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """本地动作训练；恢复时 CLI 以保存配置为默认，再应用显式参数。"""

    # 本地 NPZ 动作路径；恢复时可从检查点读取。
    motion_file: Path | None = None
    # 并行环境数；CPU 验收应显式缩小。
    num_envs: int = 2048
    # 仿真与网络设备，例如 cuda:0 或 cpu。
    device: str = "cuda:0"
    # 训练和仿真的随机种子。
    seed: int = 0
    # 训练停止时的总迭代号，包含恢复前的迭代。
    max_iterations: int = 1_000_000
    # 每个环境最多保留的时间槽数。
    buffer_capacity: int = 1000
    # 监督重用比例，训练次数的小数余额跨迭代累计。
    replay_ratio: float = 16
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    wm: WorldModelConfig = field(default_factory=WorldModelConfig)
    policy: PolicyTrainConfig = field(default_factory=PolicyTrainConfig)
    # 编译 rollout 与损失函数。
    compile: bool = True
    # 默认 Inductor；aot_eager 可用于 CPU 编译图验收。
    compile_backend: Literal["inductor", "aot_eager", "eager"] = "inductor"
    # 每隔多少次迭代打印一次标量。
    log_interval: int = 10
    # 检查点周期，结束时也保存一次。
    save_interval: int = 1000
    # 检查点父目录，各迭代使用独立子目录。
    output: Path = Path("runs/nanomimic")
    # 从包含四个检查点文件的目录恢复。
    resume: Path | None = None
    # 可选 W&B 项目，仅发送标量；默认不连接。
    wandb_project: str | None = None

    def validate(self) -> None:
        if self.motion_file is None:
            raise ValueError("训练需要 --motion-file 本地动作路径，或 --resume 检查点目录")
        integers = (
            self.num_envs,
            self.max_iterations,
            self.log_interval,
            self.save_interval,
            self.wm.batch,
            self.wm.horizon,
            self.wm.num_batches,
            self.policy.batch,
            self.policy.horizon,
            self.actor.history_length,
            self.policy.critic_epochs,
            self.policy.critic_minibatches,
        )
        if any(value < 1 for value in integers):
            raise ValueError("环境数、迭代数、批量、步数与间隔必须为正整数")
        if self.buffer_capacity <= self.wm.horizon:
            raise ValueError("缓冲容量必须至少为世界模型 horizon+1")
        if not 0 <= self.policy.gamma < 1 or not 0 <= self.policy.lam <= 1:
            raise ValueError("gamma 必须在 [0, 1)，lambda 必须在 [0, 1]")
        if self.policy.batch * self.policy.horizon < self.policy.critic_minibatches:
            raise ValueError("想象样本数不能少于 Critic minibatch 数")
        if self.replay_ratio <= 0 or min(self.wm.lr, self.policy.lr, self.policy.grad_clip) <= 0:
            raise ValueError("回放比率、学习率和梯度裁剪值必须大于零")
        if self.policy.exploration_noise < 0:
            raise ValueError("探索噪声不能为负")


def config_dict(cfg: TrainConfig) -> dict[str, Any]:
    return {"schema_version": 1, **json.loads(json.dumps(asdict(cfg), default=str))}


def load_config(directory: Path) -> TrainConfig:
    data = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if data.pop("schema_version", None) != 1:
        raise ValueError("不支持的检查点 schema_version")
    for name, cls in (
        ("dynamics", DynamicsConfig),
        ("actor", ActorConfig),
        ("critic", CriticConfig),
        ("wm", WorldModelConfig),
        ("policy", PolicyTrainConfig),
    ):
        values = data[name]
        if "hidden" in values:
            values["hidden"] = tuple(values["hidden"])
        data[name] = cls(**values)
    for name in ("motion_file", "output", "resume"):
        if data[name] is not None:
            data[name] = Path(data[name])
    return TrainConfig(**data)


def resolve_config(cfg: TrainConfig) -> TrainConfig:
    cfg.validate()
    assert cfg.motion_file is not None
    return replace(cfg, motion_file=cfg.motion_file.resolve(), output=cfg.output.resolve())

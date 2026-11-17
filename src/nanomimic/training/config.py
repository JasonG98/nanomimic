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
from nanomimic.runtime.paths import resolve_checkpoint


@dataclass(frozen=True)
class WorldModelConfig:
    #: 世界模型优化器的学习率。
    lr: float = 2e-4
    #: 每次世界模型梯度更新采样的序列窗口数。
    batch: int = 2048
    #: 每个窗口连续预测的控制步数；采样需额外一帧作为起点。
    horizon: int = 8
    #: 每轮世界模型拟合的批次数，每批执行一次梯度更新。
    num_batches: int = 10


@dataclass(frozen=True)
class PolicyTrainConfig:
    #: Actor 与 Critic 各自优化器的学习率。
    lr: float = 2e-4
    #: 每轮策略改进采样的想象起点数。
    batch: int = 2048
    #: 从每个起点在世界模型内连续想象的控制步数。
    horizon: int = 32
    #: 动作缩放前添加的高斯噪声标准差；采集与想象共用。
    exploration_noise: float = 0.1
    #: 每个控制步的回报折扣系数，范围 [0, 1)。
    gamma: float = 0.99
    #: GAE 的时间递推系数；越大越依赖后续多步回报，范围 [0, 1]。
    lam: float = 0.95
    #: Actor/Critic 中 Adam 优化器的二阶矩指数衰减系数。
    beta2: float = 0.95
    #: Actor/Critic 先将梯度范数裁到 1，再将各元素裁到正负该值内。
    grad_clip: float = 0.1
    #: 每轮策略改进中，Critic 对本轮想象数据重复训练的轮数。
    critic_epochs: int = 5
    #: Critic 每个训练轮次把本轮想象数据划分的批次数。
    critic_minibatches: int = 4
    #: 逐步启用梯度检查点，以重算中间激活换取较低显存占用。
    checkpoint: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """本地动作训练；恢复时 CLI 以保存配置为默认，再应用显式参数。"""

    #: 本地 NPZ 动作路径；恢复时可从检查点读取。
    motion_file: Path | None = None
    #: 并行环境数；CPU 验收应显式缩小。
    num_envs: int = 4096
    #: 仿真与网络设备，例如 cuda:0 或 cpu。
    device: str = "cuda:0"
    #: 训练和仿真的随机种子。
    seed: int = 0
    #: 训练停止时的总迭代号，包含恢复前的迭代。
    max_iterations: int = 100_000
    #: 每个环境最多保留的时间槽数。
    buffer_capacity: int = 1000
    #: 世界模型训练与新采集数据的样本数之比；不足一轮的训练额度累计。
    replay_ratio: float = 16
    #: 世界模型的网络结构。
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    #: Actor 的网络结构与观测历史。
    actor: ActorConfig = field(default_factory=ActorConfig)
    #: Critic 的网络结构。
    critic: CriticConfig = field(default_factory=CriticConfig)
    #: 世界模型的监督训练参数。
    wm: WorldModelConfig = field(default_factory=WorldModelConfig)
    #: Actor 与 Critic 的想象训练参数。
    policy: PolicyTrainConfig = field(default_factory=PolicyTrainConfig)
    #: 编译 rollout 与损失函数。
    compile: bool = True
    #: 默认 Inductor；aot_eager 可用于 CPU 编译图验收。
    compile_backend: Literal["inductor", "aot_eager", "eager"] = "inductor"
    #: 每隔多少次迭代打印一次标量。
    log_interval: int = 100
    #: 检查点与指标曲线的更新周期，结束时也保存一次。
    save_interval: int = 1000
    #: 输出根目录；时间戳运行内含指标、曲线及 checkpoints、evaluations、exports 子目录。
    output: Path = Path("runs/nanomimic")
    #: 从包含四个检查点文件的目录恢复。
    resume: Path | None = None
    #: 可选 W&B 项目，仅发送标量；默认不连接。
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
    directory = resolve_checkpoint(directory)
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
    return replace(
        cfg,
        motion_file=cfg.motion_file.resolve(),
        output=cfg.output.resolve(),
        resume=None if cfg.resume is None else resolve_checkpoint(cfg.resume).resolve(),
    )

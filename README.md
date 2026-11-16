# nanomimic

用可微世界模型在「想象」中反向传播，训练人形（Unitree G1）全身动作跟踪策略，并导出 ONNX 供真机部署。

- 开发规格：[docs/SPEC.md](docs/SPEC.md)
- 实测与决策记录：[docs/NOTES.md](docs/NOTES.md)
- 开发工作方式：[AGENTS.md](AGENTS.md)

## 代码结构

```text
src/nanomimic/
├── common/     数据契约、旋转数学与网络构件
├── models/     世界模型与 Actor / Critic
├── rollout/    传感器、回放缓冲、目标与可微想象
├── training/   训练更新、排程、配置、优化器与检查点
├── runtime/    仿真适配与动作校验，后续接入导出和查看器
└── cli.py      train / play / export 命令入口
```

目标窗口、奖励和软终止集中在 `rollout/imagine.py`，标量日志集中在
`training/train.py`。完整文件清单和模块契约见 [开发规格](docs/SPEC.md#2-仓库结构)。
`runtime/export.py` 与 `runtime/viewer.py` 分别留给 T8、T9，当前尚未实现。

## 开发检查

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

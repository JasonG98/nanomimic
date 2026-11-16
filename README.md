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
├── runtime/    仿真适配、动作校验、ONNX 导出和闭环虚影查看器
└── cli.py      train / play / export 命令入口
```

目标窗口、奖励和软终止集中在 `rollout/imagine.py`，标量日志集中在
`training/train.py`。完整文件清单和模块契约见 [开发规格](docs/SPEC.md#2-仓库结构)。

## 安装与训练

使用 Python 3.13 与 uv；GPU 训练面向 Linux、NVIDIA RTX 5060 Ti 16GB。
`uv sync` 在 Linux 安装 PyTorch CUDA 13.2 构建，在 Mac 安装 CPU/MPS 构建。
T1、T6 GPU、T7 的训练效果及 T8 最终 sim2sim 仍待 GPU 实测；详情见 NOTES。

```bash
uv sync
uv run nanomimic train --motion-file /绝对路径/lafan_run1_subject5.npz \
  --device cuda:0 --output runs/g1
```

动作必须是本地 NPZ，包含 joint_pos、joint_vel 和四个 body 世界系状态数组；
可选的 fps 必须等于 50，motion_ends 使用片段右开端点。形状、帧率和有限性在加载时校验，
数据顺序须匹配发布版 G1 资产。`tests/fixtures/motion.npz` 是两段合成动作，
仅用于功能验证，不能作为训练效果验收动作。

每 1000 次迭代及训练结束时保存独立检查点目录，包含 `config.json`、
`world_model.pt`、`actor.pt`、`critic.pt`，已有目录拒绝覆盖。
续训先恢复保存配置，再应用本次显式 CLI 参数，`--max-iterations` 是累计迭代数：

```bash
uv run nanomimic train --resume runs/g1/00001000 \
  --max-iterations 1000000 --output runs/g1-resume
```

在 Mac CPU 上验证训练流程时使用显式缩小的网络与批量。以下新目录已存在时需换名：

```bash
uv run nanomimic train \
  --motion-file tests/fixtures/motion.npz --device cpu --num-envs 2 \
  --max-iterations 50 --buffer-capacity 16 --replay-ratio 1 --no-compile \
  --wm.batch 2 --wm.horizon 2 --wm.num-batches 1 \
  --policy.batch 2 --policy.horizon 2 \
  --dynamics.hidden 16 16 16 --actor.hidden 16 12 8 8 --critic.hidden 16 8 \
  --actor.history-length 3 --output runs/cpu-example
uv run nanomimic train --resume runs/cpu-example/00000050 \
  --max-iterations 100 --output runs/cpu-example-resume
```

## 导出 ONNX

```bash
uv run nanomimic export --checkpoint runs/mac-cpu-resume/00000100 \
  --out outputs/g1.onnx --verify
```

导出使用 CPU，加载检查点配置对应的原动作文件；移动检查点到其他机器时，
需确保 `config.json` 的 `motion_file` 指向有效本地路径。最多支持 15000 帧，
Actor、归一化和六个动作库数组均嵌入单个 ONNX 文件。
`--verify` 用 ONNX Runtime CPU 对照 PyTorch Actor，要求动作最大绝对误差小于
`1e-4`，同时检查动作库输出与时间钳位；此命令只做数值验收。
完整动作闭环且不摔倒的 sim2sim，仍须在 T7 得到合格策略后另行验证。

部署契约固定批量为 1、fp32、opset 18；H 为检查点的历史长度，G1 的 J=29。

| 输入/输出 | 形状与含义 |
|---|---|
| `obs` | `[1, H×93+640]`：term 主序的 proprio 历史，再拼 actor_goal |
| `time_step` | `[1, 1]`：帧索引，图内转整数并钳到 `[0, 总帧数−1]` |
| `actions` | `[1, 29]`：Actor 原始动作，缩放见元数据 `action_scale`，偏置为默认关节姿态 |
| `joint_pos`、`joint_vel` | 各 `[1, 29]`，当前参考帧 |
| `body_pos_w`、`body_lin_vel_w`、`body_ang_vel_w` | 各 `[1, 跟踪 body 数, 3]`，世界坐标系 |
| `body_quat_w` | `[1, 跟踪 body 数, 4]`，wxyz |

proprio 的五个 term 依次是投影重力、基座角速度、相对关节位置、关节速度、上一动作。
每个 term 内按最旧到最新排列 H 帧，图内转为 Actor 的帧主序。
actor_goal 为 10 帧 `[joint_pos, joint_vel]` 展平后拼 10 帧相对朝向 rot6d，
目标间隔 5 控制步并钳到当前 clip 末帧，不含相对位置；历史与目标由调用方维护。
`time_step` 仅查询嵌入动作库，不替调用方合成目标。

## 回放与世界模型虚影

```bash
uv run nanomimic play --checkpoint runs/mac-cpu-resume/00000100 --viewer native --device cpu
uv run nanomimic play --checkpoint runs/mac-cpu-resume/00000100 --viewer viser --device cpu
```

`native` 打开 MuJoCo 原生窗口；macOS 自动经 `mjpython` 启动并处理 uv 的动态库路径。
`viser` 在终端打印浏览器地址，通常为 `http://localhost:8080`，Ctrl+C 结束。
原生窗口用空格暂停/继续、右箭头单步、Enter 重置；Viser 用相应按钮操作。
绿色参考虚影由 mjlab 提供，紫色虚影是 env 0 的世界模型闭环预测。
两侧 Actor 分别读取真实与预测历史和目标，预测侧使用真实的随机化上下文。

虚影每回合结束后重新对齐，默认周期是参考配置的 1000000 步。
用较短周期和有限步数验收回放功能：

```bash
uv run nanomimic play --checkpoint runs/mac-cpu-resume/00000100 \
  --viewer native --device cpu --num-steps-rollout 5 --num-steps 120
uv run nanomimic play --checkpoint runs/mac-cpu-resume/00000100 \
  --viewer viser --device cpu --num-steps-rollout 5 --num-steps 120
```

现有 Mac 检查点仅训练过 100 次小网络迭代，会摔倒并自动重置；
可用它验证回放、虚影与重新对齐，不能用于训练效果或最终 sim2sim 验收。

## 开发检查

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

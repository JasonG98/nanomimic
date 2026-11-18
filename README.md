# nanomimic

用可微世界模型在「想象」中反向传播，训练人形（Unitree G1）全身动作跟踪策略，并导出 ONNX 供真机部署。

- [图解 Wiki](docs/wiki/README.md) · 从一段 G1 动作看懂训练闭环、代码架构和效果评估。
- 开发规格：[docs/SPEC.md](docs/SPEC.md)
- [近期决策](docs/NOTES.md) · [历史决策与实测归档](docs/NOTES_ARCHIVE.md)
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
T1 与 T6 已通过 GPU 验收。T7 已开始用真实动作训练，收敛效果和 T8 最终 sim2sim 仍待验收。[验收记录与待办](docs/NOTES_ARCHIVE.md#当前状态与待办)

```bash
uv sync
uv run python tools/make_motion.py --list           # 查看 40 段片段名与本地状态
uv run python tools/make_motion.py --all            # 下载并转换 LAFAN1 全部 40 段
uv run nanomimic train --motion-file data/cache/lafan1-g1-v1/all.npz \
  --device cuda:0 --output runs/g1
```

默认训练规模为 **4096 环境、100000 次迭代**，其余超参见 SPEC 第 7 节。
动作必须是本地 NPZ，包含 joint_pos、joint_vel 和四个 body 世界系状态数组；
可选的 fps 必须等于 50，motion_ends 使用片段右开端点。形状、帧率和有限性在加载时校验，
数据顺序须匹配发布版 G1 资产。`tools/make_motion.py` 使用以下目录，
并把所选片段按命令行顺序合并成带 `motion_ends` 的缓存：

```text
data/
├── raw/lafan1/g1/                    # 下载的原始 CSV
├── processed/lafan1-g1-v1/
│   ├── clips/                       # 每段一个 NPZ
│   └── legacy/                      # 迁移保留的早期单段文件
├── cache/lafan1-g1-v1/all.npz         # 可重新生成的合并缓存
└── migrations/                      # 旧目录迁移的路径记录
```

可用 `--raw-dir`、`--out-dir`、`--merged` 显式指定这三类路径。
`--list` 只读本地文件、列出片段名与 CSV / NPZ 是否就绪，不访问网络；
也可只转换指定片段，如 `uv run python tools/make_motion.py run1_subject5`。
合并文件总帧数远超 15000，只用于训练；ONNX 导出要求动作库不超过 15000 帧，
需用单段文件或另行裁剪。`tests/fixtures/motion.npz` 是两段合成动作，
仅用于功能验证，不能作为训练效果验收动作。

每次启动在 `--output` 下创建独立的北京时间戳目录，例如
`runs/g1/20261117_153000/`，同一秒重名时自动加序号；终端会打印实际目录。
训练每 **100 次迭代**打印指标，每轮完整指标写入该目录内的 `metrics.jsonl`，
降低打印频率不会丢失曲线数据。每 **1000 次迭代**及正常训练结束时保存检查点，
并覆盖更新 `training-curves.png` 与 `training-curves.svg`。
曲线浅线为逐轮原始点、深线为向后看的 21 点均值；未执行训练的轮次不补损失。
每个检查点在 `checkpoints/` 下使用迭代号子目录，包含 `config.json`、`world_model.pt`、`actor.pt`、
`critic.pt`，已有检查点拒绝覆盖：

```text
runs/g1/20261117_153000/
├── config.json                      # 启动时保存最终生效的配置
├── run.json                         # 目录版本、创建时间和续训来源
├── metrics.jsonl
├── training-curves.png
├── training-curves.svg
├── checkpoints/
│   ├── index.json                   # 最新检查点位置，不复制模型
│   ├── 00001000/
│   └── 00002000/
├── evaluations/                     # 完整动作评估 JSON
└── exports/                         # ONNX 导出
```

续训先恢复保存配置，再应用本次显式 CLI 参数，`--max-iterations` 是累计迭代数：

```bash
uv run nanomimic train --resume runs/g1/20261117_153000/checkpoints/00001000 \
  --max-iterations 100000 --output runs/g1-resume
```

续训会创建新的时间戳目录；省略 `--output` 时沿用检查点配置中的输出根目录。
新目录的指标与曲线只包含本次续训的迭代，原运行记录保留。

在 Mac CPU 上验证训练流程时使用显式缩小的网络与批量：

```bash
uv run nanomimic train \
  --motion-file tests/fixtures/motion.npz --device cpu --num-envs 2 \
  --max-iterations 50 --buffer-capacity 16 --replay-ratio 1 --no-compile \
  --wm.batch 2 --wm.horizon 2 --wm.num-batches 1 \
  --policy.batch 2 --policy.horizon 2 \
  --dynamics.hidden 16 16 16 --actor.hidden 16 12 8 8 --critic.hidden 16 8 \
  --actor.history-length 3 --output runs/cpu-example
# 把时间戳替换为上一条命令打印的实际目录。
uv run nanomimic train --resume runs/cpu-example/20261117_153000/checkpoints/00000050 \
  --max-iterations 100 --output runs/cpu-example-resume
```

## 完整动作效果评估

```bash
uv run python tools/evaluate_tracking.py \
  --checkpoint runs/t7-lafan/checkpoints/00001000 --num-envs 64 --device cuda:0 --seed 0
```

评估仅接受单段动作，从第 0 帧连续跑到末帧。关闭初态扰动和动作探索噪声，
保留脚摩擦、质心偏移、编码器偏置与传感器噪声；策略和归一化均不更新。
评估延长 10 秒的训练超时，关闭自动重置；失败状态先计分，之后仅重置失败行以继续
其他尝试，重置后的数据不进入统计。成功必须在首次尝试中走完整段，提前超时也算失败。
省略 `--out` 时保存到运行目录的 `evaluations/eval-迭代号-seed种子.json`；显式路径优先。

JSON 同时记录完整动作成功率、平均存活时间、完成比例、尝试期间误差与仅成功尝试的
误差。`error_body_pos` 是重锚定后 14 个跟踪 body 的欧氏距离均值（m），
`error_body_rot` 是旋转误差角均值（rad），关节误差为整个关节向量的 L2 范数。
尝试期间误差按实际控制步加权，包含失败的最后一步；低误差本身不能证明完整跟踪成功。
比较检查点应保持动作 SHA-256、种子、尝试数和评估协议一致。当前缺少同段动作的
SuperMimic 参考结果。取得可比结果后，将待用户确认的效果阈值写入[决策记录](docs/NOTES.md)。

从本轮日志绘制损失、真实奖励、完整动作评估及整卡资源曲线：

```bash
uv run python tools/plot_training.py --log runs/t7-lafan-observation/train.log \
  --evaluations runs/t7-lafan --vram runs/t7-lafan-observation/vram.csv \
  --out outputs/t7-curves
```

输出 PNG 与 SVG；重复执行会更新图片。浅线是原始日志，深线是 21 点滑动均值；
世界模型和 Critic 损失用对数轴。Actor 损失是含末端价值的负想象回报，结合真实奖励
和完整动作评估判断效果。资源曲线包含桌面、训练及并行评估，不单独代表训练进程占用。

## 导出 ONNX

```bash
uv run nanomimic export --checkpoint runs/mac-cpu-resume/checkpoints/00000100 --verify
```

导出使用 CPU，加载检查点配置对应的原动作文件；移动检查点到其他机器时，
需确保 `config.json` 的 `motion_file` 指向有效本地路径。最多支持 15000 帧，
Actor、归一化和六个动作库数组均嵌入单个 ONNX 文件。
省略 `--out` 时写入运行目录的 `exports/actor-迭代号.onnx`；仍可显式指定任意输出路径。
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
uv run nanomimic play --checkpoint runs/mac-cpu-resume/checkpoints/00000100 --viewer native --device cpu
uv run nanomimic play --checkpoint runs/mac-cpu-resume/checkpoints/00000100 --viewer viser --device cpu
```

`native` 打开 MuJoCo 原生窗口；macOS 自动经 `mjpython` 启动并处理 uv 的动态库路径。
`viser` 在终端打印浏览器地址，通常为 `http://localhost:8080`，Ctrl+C 结束。
原生窗口用空格暂停/继续、右箭头单步、Enter 重置；Viser 用相应按钮操作。
绿色参考虚影由 mjlab 提供，紫色虚影是 env 0 的世界模型闭环预测。
两侧 Actor 分别读取真实与预测历史和目标，预测侧使用真实的随机化上下文。

虚影每回合结束后重新对齐，默认周期是参考配置的 1000000 步。
用较短周期和有限步数验收回放功能：

```bash
uv run nanomimic play --checkpoint runs/mac-cpu-resume/checkpoints/00000100 \
  --viewer native --device cpu --num-steps-rollout 5 --num-steps 120
uv run nanomimic play --checkpoint runs/mac-cpu-resume/checkpoints/00000100 \
  --viewer viser --device cpu --num-steps-rollout 5 --num-steps 120
```

现有 Mac 检查点仅训练过 100 次小网络迭代，会摔倒并自动重置；
可用它验证回放、虚影与重新对齐，不能用于训练效果或最终 sim2sim 验收。

## 迁移旧目录

已下载的数据与旧训练产物可以直接移动，不需要重新转换或训练：

```bash
uv run python tools/migrate_layout.py          # 只展示计划
uv run python tools/migrate_layout.py --apply  # 执行移动并保存路径记录
```

迁移保留全部检查点，不改写其配置或权重。旧的平铺检查点参数，如
`--checkpoint runs/t7-lafan/00001000`，仍能定位到迁移后的文件。
旧运行名称保留；若一个实验根目录已混有新时间戳运行，其旧平铺产物收进 `legacy/`。
旧动作路径保留相对软链接，因此旧检查点仍可找到原动作文件。
迁移工具会先检查所有目标冲突，拒绝覆盖；同一文件系统内移动后逐文件核对身份与大小。
`data/migrations/` 保存执行前的路径计划及完成记录，重复执行不会再次移动已整理文件。

## 开发检查

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

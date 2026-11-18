# 安装与复现

在仓库根目录运行以下命令。正式训练使用 Linux 和 NVIDIA GPU。Linux 环境会安装 CUDA 13.2 的 PyTorch 构建，macOS 可用 CPU 验证训练和回放流程。

## 安装

首次使用发布版时，克隆对应标签，再用 uv 同步锁定的 Python 3.13 依赖。[uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone --branch v0.1.0 https://github.com/JasonG98/nanomimic.git
cd nanomimic
uv sync --locked
uv run nanomimic --help
uv run python -c 'import torch; print(torch.cuda.is_available())'
```

最后一条输出 `True` 表示 PyTorch 可以使用 CUDA。

## 使用预训练权重

v0.1.0 提供在 40 段 LAFAN1 动作上训练 100000 轮的 G1 策略。只想试跑一段舞步，可以直接执行首页的 ONNX 命令。需要回放世界模型或续训时，下载并解压完整检查点。[单段试跑](../README.md#试用预训练策略) · [发布页](https://github.com/JasonG98/nanomimic/releases/tag/v0.1.0)

```bash
RELEASE=https://github.com/JasonG98/nanomimic/releases/download/v0.1.0
DEST=runs/pretrained/v0.1.0
mkdir -p "$DEST"
(
  cd "$DEST" || exit 1
  for file in nanomimic-g1-100k.tar.gz nanomimic-g1-100k.onnx evaluation.json SHA256SUMS; do
    curl -fLO "$RELEASE/$file" || exit 1
  done
  shasum -a 256 -c SHA256SUMS && tar -xzf nanomimic-g1-100k.tar.gz
)
CHECKPOINT="$DEST/checkpoints/00100000"
uv run nanomimic export --checkpoint "$CHECKPOINT" --verify
```

解压后有 `actor.pt`、`critic.pt`、`world_model.pt` 和 `config.json`。上面的 CPU 数值自检无需参考动作。要在浏览器里看策略和世界模型的回放，从仓库根目录生成训练时使用的 40 段动作库，再启动 Viser。

```bash
uv run python tools/make_motion.py --all
uv run nanomimic play --checkpoint "$CHECKPOINT" --viewer viser
```

继续这次训练时，把累计迭代上限设到 100000 以上。下面会另外训练 10000 轮并创建新的运行目录。

```bash
uv run nanomimic train --resume "$CHECKPOINT" --max-iterations 110000
```

## CPU 快速验证

用仓库内的合成动作和小网络跑 12 轮，检查训练和保存流程。

```bash
uv run nanomimic train \
  --motion-file tests/fixtures/motion.npz --device cpu --num-envs 2 \
  --max-iterations 12 --buffer-capacity 16 --replay-ratio 1 --no-compile \
  --wm.batch 2 --wm.horizon 2 --wm.num-batches 1 \
  --policy.batch 2 --policy.horizon 2 \
  --dynamics.hidden 16 16 16 --actor.hidden 16 12 8 8 --critic.hidden 16 8 \
  --actor.history-length 3 --output runs/cpu-smoke
```

终端会打印运行目录与最终检查点路径。这个检查点也可以用来试回放与导出。

## 用 LAFAN1 训练

先用一段跑步动作，训练完成后可以评估整段跟踪并导出 ONNX。

```bash
uv run python tools/make_motion.py run1_subject5
uv run nanomimic train \
  --motion-file data/processed/lafan1-g1-v1/clips/run1_subject5.npz \
  --device cuda:0 --output runs/g1-run
```

默认配置跑 4,096 个并行环境、最多 100,000 轮。初次检查 GPU 训练流程，可以给训练命令加上 `--num-envs 2048 --max-iterations 50`。显存实测和完整超参见[实测归档](NOTES_ARCHIVE.md#gpu-实测与编译边界2026-11-16)和[开发规格](SPEC.md#7-默认配置)。

训练整套 40 段 G1 动作时，用合并文件。

```bash
uv run python tools/make_motion.py --all
uv run nanomimic train \
  --motion-file data/cache/lafan1-g1-v1/all.npz --device cuda:0 --output runs/g1
```

原始 CSV 在 `data/raw/lafan1/g1/`，每段转换的 NPZ 在 `data/processed/lafan1-g1-v1/clips/`，合并文件在 `data/cache/lafan1-g1-v1/all.npz`。用 `--list` 查看片段及本地状态，或用 `--raw-dir`、`--out-dir`、`--merged` 指定路径。自备的 NPZ 须按 G1 资产顺序保存关节和 body 状态，帧率为 50 Hz，`motion_ends` 是各片段的右开端点。[数据格式](SPEC.md#41-runtimesimpy)

### 找到检查点并续训

每次运行会在 `--output` 下按北京时间创建带时间戳的目录。目录中的 `config.json` 保存实际配置，`metrics.jsonl` 每轮记录指标。每 100 轮打印一次，每 1,000 轮以及正常结束时保存检查点并更新 PNG 和 SVG 曲线。

```text
runs/g1-run/<时间戳>/
├── config.json
├── run.json
├── metrics.jsonl
├── training-curves.png
├── training-curves.svg
├── checkpoints/
│   ├── index.json
│   └── 00001000/
├── evaluations/
└── exports/
```

续训、回放、评估和导出都使用终端打印的 `<运行目录>/checkpoints/<迭代号>`。最新迭代号也可从 `checkpoints/index.json` 的 `latest` 字段读取。目录内包含世界模型、Actor、Critic 和配置。评估和导出默认把产物存到同一运行的 `evaluations/` 和 `exports/`。

```bash
# 把示例时间戳和迭代号换成实际值。
CHECKPOINT=runs/g1-run/20261118_003440/checkpoints/00001000
uv run nanomimic train --resume "$CHECKPOINT" --max-iterations 100000
```

续训在输出根目录下创建新运行。它先加载保存的配置，再应用本次明确传入的参数。`--max-iterations` 是累计迭代上限，新的指标文件从续训开始记录。

## 回放、评估与导出

### 回放

```bash
uv run nanomimic play --checkpoint "$CHECKPOINT" --viewer native
uv run nanomimic play --checkpoint "$CHECKPOINT" --viewer viser
```

原生窗口用空格暂停、右箭头单步、Enter 重置。Viser 会打印浏览器地址。绿色虚影是参考动作，紫色虚影是世界模型的闭环预测。回合结束时会自动对齐虚影，也可用 `--num-steps-rollout 50` 每 50 步对齐。用 `--device cpu --num-steps 120` 可做有限步回放。

### 评估整段动作

```bash
uv run python tools/evaluate_tracking.py \
  --checkpoint "$CHECKPOINT" --num-envs 64 --device cuda:0 --seed 0
```

评估需要单段动作。从第 0 帧开始，关闭初态扰动和探索噪声，保留域随机化与传感器噪声。每次尝试连续跟踪到末帧才计为成功。结果包含成功率、存活时间、完成比例和误差。对比时使用同一动作哈希、种子、尝试数和协议。[误差口径与实测](NOTES_ARCHIVE.md#评估协议)

已有训练日志、评估 JSON 和整卡显存 CSV 时，可用 `uv run python tools/plot_training.py --help` 查看合并绘图的选项。

### 导出 Actor 并评估 ONNX

```bash
ONNX=/tmp/nanomimic-actor.onnx
uv run nanomimic export --checkpoint "$CHECKPOINT" --out "$ONNX" --verify
uv run python tools/evaluate_tracking.py --onnx "$ONNX" \
  --motion-file data/processed/lafan1-g1-v1/clips/run1_subject5.npz \
  --num-envs 1 --device cpu --seed 0
```

导出只需检查点的 `config.json` 和 `actor.pt`，包括用合并动作库训练的检查点。省略 `--out` 时，ONNX 写入运行目录的 `exports/actor-迭代号.onnx`。`--verify` 在 CPU 比较 ONNX Runtime 和 PyTorch Actor 的输出，要求最大绝对误差小于 `1e-4`。

闭环评估通过 `--motion-file` 选择外置单段 NPZ。更换为其他片段时可以复用同一个 ONNX。评估时 Actor 始终在 CPU 推理，仿真可通过 `--device cuda:0` 在 GPU 运行。

### 部署观测与动作

ONNX 固定批量为 1、fp32、opset 18。它接收一个 `obs[1, 93H+640]`，输出 `actions[1, 29]`。从模型元数据读取 `history_length`、`control_dt` 与 `joint_names`，按模型的关节顺序组织输入和输出。数字与数组字段用 JSON 解码。[完整接口](SPEC.md#48-runtimeexportpy)

每个控制步使用投影重力、训练所用 IMU 坐标系的角速度、关节位置减 `default_joint_pos`、关节速度和上一帧的原始 Actor 动作构造一帧观测。每项按最旧到最新拼接 `H` 帧，然后附上参考目标。首次重置可用首帧填满历史，上一动作置零。

从当前动作帧起，每隔 5 帧取一个目标，共取 10 帧。越过当前片段右开端点 `motion_ends` 时取该片段最后一帧。将各帧的 `[joint_pos, joint_vel]` 依次展平，随后拼接相对朝向。用 `motion_body_names` 查 `anchor_body_name` 在 NPZ 中的列，将参考 `pelvis` 的世界系四元数转换成旋转矩阵。每帧计算 `R_当前pelvis.T @ R_参考pelvis`，取前两列并按行展平。目标共 640 维。

调用 `session.run(["actions"], {"obs": obs})` 后，按 `action_offset + action_scale * actions` 得到关节目标。保留原始 `actions` 供下一步观测。归一化包含在 ONNX 图里。[CPU 推理代码](../src/nanomimic/runtime/inference.py)

## 开发工具

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

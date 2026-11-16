# NOTES

记录实测数据、与参考仓库（HybridRobotics/supermimic）的差异、以及等待用户确认的事项。
按任务追加，最新内容放在最前面。

## 2026-11-16 · T6 编译耗时复查：已确认大图规模，未确认框架缺陷

- 此前完成的是编译边界调整和 T6 验收，并未定位首次半小时中的具体编译器热点。
  「编译修复」应理解为调整不合适的编译范围，不能据此宣称已修复或已排除
  PyTorch 内部 BUG，也不能把改后 178 秒与原路径的中断时间当作受控速度比。
- 复查原运行留下的 Inductor 产物：世界模型前向 / 反向源码在 21:50:16–21:50:30
  生成；想象前向模块在 **22:08:02** 生成，大小 **9,801,891 字节、96,333 行**，
  定义 **338** 个 Triton kernel，含 **6681** 个 Triton 调用点。
  22:08:20 已生成 Actor 回报损失的前向 / 反向模块，说明原运行确有编译及执行进展。
  截至原运行中断，未找到对应的整段想象反向模块；缺少原阶段 trace 和调用栈，
  不能进一步断言具体停在反向编译的哪个 pass。产物文件时间差不是精确的分阶段计时。
- 新增一次性 CPU 诊断：使用默认网络、B=2、原有 checkpoint，整段编译边界保持原样；
  自定义 backend 只统计 Dynamo FX 图，再返回 `gm.forward`，**不运行 Inductor**。
  计入 checkpoint 子图，1 / 4 / 32 步的 FX 节点分别为 **4224 / 16149 / 127449**，
  每档均只有一个编译区域。这直接确认时间循环被展开成单个大图；每步的 30 body FK、
  目标、网络、奖励、软终止和传感器计算都会被重复纳入。
- 上述诊断只用于图结构归因：B=2 不替代默认 GPU 验收；图捕获加 CPU 前向的耗时
  不代表 GPU 完整编译耗时。脚本和结果位于 **`runs/t6-compile-audit/`**，不入 Git。
- 最终逐步方案的原始日志有 Dynamo 编译汇总 **119.3145 秒**，其中代码生成指标
  **61.8799 秒**、融合调度 **23.5295 秒**、kernel 自调优 **35.6922 秒**。
  指标存在父子嵌套，不能相加。这些是最终方案的数据，不能用于补算原半小时的阶段分布。
- 当前有证据的工程结论：**原编译边界把 32 步重复计算展开成约 12.7 万节点，
  编译范围过大；使用逐步区域编译可完成原验收。** Python 循环展开导致大图和编译
  开销，是 [PyTorch 控制流文档](https://docs.pytorch.org/docs/main/higher_order_ops/index.html)
  已说明的限制；[区域编译](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
  是官方提供的应对方式。不能把半小时简单视为模型必须付出的固定成本。
- **仍未确认**：原路径是否还触发特定的 Inductor 性能缺陷、具体最慢 pass、
  相同空缓存下的整段 / 逐步耗时比。撤回的 `nested_compile_region` 方案有独立的
  4 步梯度对照失败；尚未区分随机数处理、调用方式与框架缺陷，不能称为已确诊的
  PyTorch 正确性 BUG，也不能用它解释更早发生的整段编译耗时。
- 本次只做归因复查和记录修正，没有再次修改训练源码、默认配置或测试，
  没有推进 T7，没有提交或推送；T6 既有 50 次 GPU 验收及 61 项测试结论不变。

## 2026-11-16 · T6 GPU 50 次迭代（验收通过）

### 实测结果

- RTX 5060 Ti 16GB，默认 **2048 环境**、每环境 **1000 步缓冲**、replay_ratio=16；
  世界模型保持 3×2048、batch=2048、horizon=8、每轮 10 个 batch；策略保持
  batch=2048、想象 32 步、逐步 checkpoint；Actor / Critic、优化器与学习率均不缩小。
  编译保持开启，后端为 **Inductor**，参数 fp32、矩阵乘 TF32。
- 使用 `tests/fixtures/motion.npz`、seed=0，完成 **50 次迭代**，退出码 **0**。
  检查点为 **`runs/t6-gpu/00000050`**；保存配置与只覆盖动作路径、迭代数、输出路径
  的 `TrainConfig` 默认值逐项相等。累计 **10 轮 fit / improve**，排程余额为 0；
  世界模型 / Actor / Critic 的 Adam 更新计数分别为 **100 / 10 / 200**。
- 全程没有非有限损失或梯度异常。CPU 加载三个检查点，模型与归一化共 **74** 个
  张量、优化器共 **162** 个张量全部有限；三个检查点迭代号均为 50。

| 第 50 次迭代指标 | 值 |
|---|---|
| 世界模型损失 | 0.136690 |
| 策略损失 | -1.41326 |
| Critic 损失 | 0.0411164 |
| 仿真奖励 | 0.0605439 |
| 重置比例 | 0.019043 |
| PyTorch 已分配 / 保留显存峰值 | 2.98491 / 3.43094 GB |
| **整卡采样峰值** | **4610 MiB = 4.502 GiB = 4.833935 GB < 15 GB** |

- 独立终端每 **200 ms** 采样整卡显存，共 **1165** 个样本，覆盖训练启动前至
  退出后；峰值包含 PyTorch、Warp、MuJoCo 和用户允许保留的桌面占用，未扣基线。
  最初 10 个样本均为 **618 MiB**，最后 10 个均为 **689 MiB**；桌面占用有波动。
  每秒的计算进程记录包含训练 PID 34597，以及已核实的 VS Code GPU 进程 23261、
  Chrome GPU 进程 36965，没有其他训练任务。200 ms 采样不保证捕获更短的瞬时峰值。
- 最终运行时间为北京时间 **22:36:41–22:39:40，共 178.42 秒**，包含初始化、编译、
  训练与保存；使用已有编译缓存，不把这个数字当作冷启动耗时或稳定训练吞吐。
  增加的 `TORCH_LOGS=dynamo` 仅用于确认编译阶段，未改训练配置。
- 环境同 T1：Python 3.13.15、torch **2.15.0+cu132**、PyTorch CUDA **13.2**、
  驱动 **615.71.09** / CUDA UMD **13.4**；mjlab 1.6.1、MuJoCo / MuJoCo Warp
  3.11.0、Warp 1.19.0（其启动日志报告 CUDA Toolkit / Driver 13.4）。

### 编译修复与回归验证

- 原实现把整个 32 步想象循环交给 `torch.compile` 展开；首次运行约 30 分钟仍未
  输出第 10 次迭代。SIGINT 未及时退出，中断阶段观测到进程 RSS 约 18 GiB，
  随后只对自建训练进程发送 SIGTERM，退出码 143；该次没有检查点，不计为验收通过。
- 改为编译 `imagine_step`，由原 Python 时间循环逐步调用；每步仍使用原有
  checkpoint，全程保留跨步梯度、RNG 重算语义、目标 / 奖励 / 软终止公式。
  世界模型 rollout 与损失、Actor / Critic 损失继续编译，没有关闭编译或改用 eager
  后端来取得 GPU 验收。该选择采用 PyTorch 文档的
  [区域编译方式](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)。
- 探索中曾尝试 `nested_compile_region`，2 步对照通过，但新增的 4 步对照发现
  梯度最大绝对差 **0.003919743**，超过原阈值；该方案及其输入复制均已撤回，
  对应 GPU 运行主动终止、不计验收。最终实现使用普通 `torch.compile`，不依赖该接口。
- 将既有策略编译对照参数化为 **2 / 4 步**，覆盖重复单步的复用、动力学冻结与
  Critic 的 5×4 次更新。最终两种步数下，编译与未编译的 Actor 梯度均满足原有
  `atol=1e-6, rtol=1e-5`；checkpoint 开关、T3/T5 对拍等既有测试全部通过。
- 最终 **ruff、格式检查（43 文件）、pyright（0 errors / warnings）、完整 pytest
  61 passed，5 warnings，24.36 秒**通过。测试警告仍为已记录的 TorchScript /
  旧 ONNX API 弃用提示；GPU 日志另有 Triton FP fusion 默认值及 SM 数量提示，
  没有 CUDA 错误或接触容量不足提示。夹具和所有误差阈值均未修改。

### 状态与复现

- **T6 GPU 验收通过；保留 `num_envs=2048`。** 本次仅调整想象 rollout 的编译
  边界并增强回归验证，没有数学、默认超参数或验收标准的 SPEC 偏离，
  无需用户确认的偏离项。此前 T6 GPU「待用户运行」记录是历史状态，以本条为准。
- 本次未推进 T7 或 T8 最终 sim2sim；50 次合成动作训练仅验收稳定性与资源，
  不代表收敛效果。下一阶段仍为 T7 的真实动作收敛训练和参考阈值比较，
  最终 sim2sim 须使用 T7 合格策略。用户已在本轮明确授权提交；推送未授权，本次不执行。
- 最终训练日志、前后进程清单、检查点 / 优化器核验、完整测试日志和统计汇总在
  **`runs/t6-gpu-observation/`**；整卡原始采样为 **`runs/t6-gpu-vram-mib.csv`**。
  前两次未通过的尝试分别保留为 `initial-*`、`nested-*`；所有运行产物沿用忽略规则。

复现（另开终端先启动采样，训练退出后 Ctrl+C；重复运行须更换输出与采样文件名）：

```bash
nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits \
  -lms 200 > runs/t6-gpu-vram-mib.csv
```

```bash
TORCH_LOGS=dynamo uv run nanomimic train \
  --motion-file tests/fixtures/motion.npz --device cuda:0 \
  --max-iterations 50 --output runs/t6-gpu
```

## 2026-11-16 · T1 RTX 5060 Ti 实测（GPU 验收通过）

### 环境与测量口径

- 本机 NVIDIA GeForce RTX 5060 Ti 16GB（sm_120）；`nvidia-smi` 报告整卡
  16311 MiB，驱动 / NVIDIA-SMI 版本 **615.71.09**，CUDA UMD **13.4**。
  PyTorch **2.15.0+cu132**、`torch.version.cuda=13.2`，CUDA 可用；Warp 启动
  日志报告 CUDA Toolkit / Driver 均为 **13.4**。这些版本分别记录，不混作同一项。
- Linux x86_64、Intel Core i7-14700KF、Python 3.13.15；mjlab 1.6.1、
  MuJoCo / MuJoCo Warp 3.11.0、Warp 1.19.0。未安装私有 fork，未改参考仓库。
- 严格沿用下方 T1 的两条 `tools/check_sim.py` 命令，同一份 96 帧合成动作、
  seed=0，每档独立进程运行 1000 个控制步，控制步长 0.02 秒。
  吞吐为 `num_envs×1000/耗时`，包含随机动作生成、形状 / 有限性 / 上下文
  检查及重置统计；计时从构造和 reset 后开始，首尾均同步 CUDA。
  未增加额外预热，未去掉检查，也不把该结果当成纯物理或训练吞吐。
- 在独立终端用 `nvidia-smi` 每 **200 ms** 采样整卡显存，从被测进程启动前
  持续到退出后，包含 PyTorch、Warp、MuJoCo 及桌面占用。另每秒记录计算进程；
  两档期间分别仅有验收进程 PID 22597 / 22831，没有其他计算任务。
- 用户明确选择保留桌面应用并计入基线；实际测量前两档基线分别为
  **404–405 MiB**、**405–406 MiB**（各取最初 10 个样本），退出后约 403–404 MiB。
  下表峰值未扣基线，属于 200 ms 采样到的最大值，不保证捕获更短的瞬时峰值。

### 实测结果

| num_envs | 控制步数 | 含检查耗时 | 环境步数/秒 | 自动重置次数 | 整卡峰值 MiB | 整卡峰值 GiB |
|---|---|---|---|---|---|---|
| 2048 | 1000 | 17.84 秒 | **114795.9** | 48238 | **948** | **0.926** |
| 4096 | 1000 | 28.81 秒 | **142176.8** | 96324 | **1247** | **1.218** |

- 两档均以退出码 0 通过，公开观测和上下文的形状、有限性、取值范围检查全部通过。
  日志没有接触容量不足或 CUDA 错误，只有已知的 `torch.jit.script` 弃用提示。
- 显存分别取得 381 / 608 个样本，采样窗口为北京时间 21:40:58–21:42:14 /
  21:42:38–21:44:40；峰值首次出现于 21:41:38.411 / 21:43:39.880。
  2048 首次运行触发 Warp 内核编译，4096 使用已有缓存；表中均为脚本原始输出，
  未扣除计时区间内的任何开销，也未重复运行取最好值。
- 原始环境信息、前后进程清单、两档日志、显存 CSV、计算进程 CSV 与
  `summary.json` 保存在 **`runs/t1-gpu/`**，沿用现有忽略规则，不入 Git。
  动作夹具 SHA-256 为
  `c4030977382668861961dddc3879404d7bbc448417d2dae4b3d94d9b963f988d`；
  夹具和验收脚本均未修改。

### 默认值结论与验收边界

- **确定保持 `num_envs=2048`**，与 SPEC 第 7 节及现有配置一致。
  4096 的环境吞吐提高 **23.85%**，整卡仿真峰值仅增加 **299 MiB**，可作为
  后续可选规模；但 T1 没有加载训练网络、优化器、回放缓冲或想象反向传播。
  按当前字段和每环境 1000 步容量计算，回放张量容量约从 **1.743 GiB**
  增至 **3.487 GiB**（仅按形状和 dtype 推算，非 GPU 实测）。因此先保留 2048
  给完整训练留出余量，本次数据不足以据此上调训练默认值。
- **无 SPEC 偏离，无需用户确认的偏离项。** T1 GPU 测量和默认值结论已取得；
  用户已在本轮明确授权提交；推送未授权，本次不执行。此前记录中的 T1「待用户在
  RTX 5060 Ti 上运行」是历史状态，以本条为准。
- 本次未运行 T6 GPU、T7 或 T8 最终 sim2sim；默认训练显存低于 15GB 的验收
  仍待 T6，T7 效果阈值仍须与参考实现比较后由用户确认，最终 sim2sim 须用
  T7 合格策略。

复现命令（每档开始前在另一终端启动采样，进程退出后 Ctrl+C；4096 档更换文件名）：

```bash
mkdir -p runs/t1-gpu
nvidia-smi --id=0 \
  --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv -lms 200 > runs/t1-gpu/2048-vram.csv
```

```bash
uv run python tools/check_sim.py --device cuda:0 --num-envs 2048 --steps 1000
uv run python tools/check_sim.py --device cuda:0 --num-envs 4096 --steps 1000
```

### 代码检查

- `uv run ruff check .`、`uv run ruff format --check .`（43 个文件）、
  `uv run pyright`（0 errors / warnings）及 `uv run pytest`
  **60 passed，5 warnings，43.04 秒**全部通过。
- 沙箱内先用项目内 `UV_CACHE_DIR` 运行检查；Pyright 未解析到虚拟环境依赖，
  改在沙箱外执行同一条 `uv run pyright` 后通过，没有改类型配置或压制错误。
  GPU 访问同样使用沙箱外执行；pytest 仍为 CPU 验证，没有调用训练 GPU 验收。
  五条测试警告均为已有的 TorchScript / 旧 ONNX 导出 API 弃用提示。
- 本次仅更新本记录，未改源码、默认配置、SPEC、测试、夹具或误差阈值。

## 2026-11-16 · T8、T9 推进安排（本次用户明确授权）

- T6 的实现与 CPU 验证之后，先连续完成 T8 的实现与 CPU 数值验收，再完成
  T9 的实现与 Mac CPU 回放实测，不等待 GPU。已同步 SPEC，保留所有原验收标准。
- T1、T6 GPU 部分、T7、T8 最终 sim2sim 继续待用户在 RTX 5060 Ti 上运行；
  最终 sim2sim 等 T7 得到合格策略后执行。试跑检查点仅验证功能，不代表训练效果。
- 已找到 `runs/mac-cpu/00000050` 和 `runs/mac-cpu-resume/00000100`：
  CPU、两环境、小网络、H=3，无需重新训练。沿用五目录结构，参考仓库只读。
- 本次 Git 提交与推送未授权，不执行。

### T8 实现与 CPU 数值实测

- 接通 `export --verify`；输入 `obs` 为 fp32 `[1, H×93+640]`，
  `time_step` 为 fp32 `[1, 1]`，图内转整数索引。历史每个 term 内最旧在前，
  目标为 580 维关节窗口加 60 维相对朝向，不含 30 维相对位置。
- 沿用参考 motion 部署图的输出名 `actions`，其余六项为 SPEC 指定的动作张量。
  ONNX 内嵌全部六个动作库常量，跟踪 body 顺序与 `body_names` 一致；固定批量 1、
  opset 18、无外部数据文件，拒绝超过 15000 帧的动作库。
- 时间上界沿用参考的全动作库末帧；额外把负时间钳到 0，避免负索引意外取末帧。
  这不改变有效时间的契约；目标窗口仍由既有实现按所属 clip 末帧钳位。
- 发布版 mjlab 的元数据函数仅在 `runtime/sim.py` 调用，维持导入隔离；
  补齐 anchor、body、两个目标 term 与历史长度，同时扩展 scale/clip 等逐 term
  数组，使其与七个观测 term 一一对应。参考导出器只扩展名称和一组历史长度。
- 使用已安装 torch 2.15 的公开 `torch.onnx.export(dynamo=False)`，与参考导出
  契约一致且不增加依赖。Context7 与本地发布版源码均已核对；该路径有第三方
  弃用警告，当前导出及 ONNX Runtime CPU 推理均可用。
- 100 次试跑检查点：96 帧，43 组随机历史/目标及边界时间的 CPU 自检通过；
  最大动作绝对误差 **5.36442e-7 < 1e-4**，六个动作库输出逐元素严格相等。
  新增 7 项测试通过；T8 阶段完整 pytest **54 passed（26.19 秒）**，
  ruff、格式检查与 pyright（0 errors）通过。

```bash
uv run nanomimic export --checkpoint runs/mac-cpu-resume/00000100 \
  --out outputs/mac-cpu/policy.onnx --verify
uv run pytest tests/test_export.py
```

### T9 实现与 Mac CPU 实测

- 接通 `play --viewer native|viser`，默认一个真实环境；只对 env 0 维护
  一条预测状态、动作时间、随机化上下文和历史。真实侧与预测侧独立计算目标、
  调用 Actor，动力学使用预测侧动作，没有把真实动作回灌成开环预测。
- 两侧统一通过既有 `add_noise` 加传感器噪声；不加训练时的动作探索噪声，
  不更新网络与归一化。虚影使用同一控制步长与 clip 末帧钳位。
- 回合结束时对齐返回的新回合状态；定期对齐只重置预测历史，保留真实侧历史。
  上一动作取真实观测的对应通道。其他环境 done 不会重置 env 0 的虚影。
  也处理 viewer 重置及 Viser 动作控件绕过 viewer 的底层重置。
- 默认 `num_steps_rollout=1000000` 取自参考 `scripts/play.py`，没有替换成
  训练想象 horizon。实测显式设为 5 或 10，便于在短运行里观察对齐。
- 两种查看器各一个薄子类，单步共享 `GhostRollout`，使用发布版公开的
  `add_ghost_mesh(qpos, model, alpha, label)`，无需参考仓库私有 ghost 接口。
  qpos 按发布版实体的地址映射写入，根位置补 env 0 世界偏移；模型为独立副本，
  `geom_rgba=[0.5, 0.5, 0.8, 0.8]`，不改变真实仿真模型。
- 发布版 native ghost 默认启用 `mjVIS_TRANSPARENT`，最终 alpha 还乘
  `model.vis.map.alpha`（默认 0.3），因此渲染 alpha 为 0.24；模型 RGBA 仍为规格值。
  Context7 与本地发布版源码核对后，绘制测试按此契约检查，误差阈值仍为 1e-7。
- macOS 原生入口自动经 `mjpython` 重启；uv 的 venv 解释器为符号链接，
  需把 `sys.base_prefix/lib` 加入启动进程的动态库查找路径。首次遇到的
  libpython 加载失败已修复，并实际验证 CLI 启动。GUI/本机端口实测需要在
  agent 沙箱外运行；普通终端直接使用下面的命令即可。
- 100 次试跑检查点、Mac CPU：native 与 viser 各 **120 步**，均为回合对齐
  **3 次**、定期对齐 **21 次**。native 延长到 **2000 步**，回合对齐 **49 次**、
  定期对齐 **170 次**，没有回放异常或非有限虚影。
- 收尾定位到 macOS native 的退出边界：发生回合重置后，Python 已打印结束且查看器
  和仿真已关闭，mjpython 仍可能停在 Cocoa 原生退出阶段。基础 mjpython、只导入
  PyTorch/MuJoCo/viewer 的对照均可退出；无回合重置的两步与含重置的 120 步形成复现。
  因此仅在 macOS native **成功完成有限步**回放后调用 `os._exit(0)`，默认交互回放
  不改。修复后含 **3 次**回合对齐的 120 步及 **2000 步**进程均实测以退出码 **0**
  正常结束；2000 步仍为回合对齐 **49 次**、定期对齐 **170 次**。
- Viser 浏览器实际观察到绿色参考虚影、紫色世界模型虚影和真实机器人，
  验证暂停、单步（步数 0→1）、Reset Environment 后虚影立即对齐。
  该次总计 **646 步**，回合对齐 **15 次**、周期对齐 **55 次**，最后 Ctrl+C 关闭。
  对齐画面保存在 `outputs/mac-cpu/viser-aligned.jpg`，产物按既有规则不入 Git。
  原生窗口已实际运行；计算机使用工具连接原生窗口超时，未取得其截图。
- 试跑策略仍会摔倒、自动重置；较长回放出现发布版 Warp 的 broadphase
  接触容量提示（建议 nconmax/naconmax 36–39）。本轮未改仿真接触容量或训练
  默认值；上述实测只证明回放、虚影和对齐功能，不构成物理精度或训练效果验收。
- 新增 6 项回放测试：独立预测输入和动作、周期对齐、其他环境 done 隔离、
  回合及手动重置、真实 MuJoCo 场景绘制与模型隔离、CLI 和非法配置。
  README 已补齐安装、GPU/CPU 训练、续训、导出契约、回放与交互说明。
- 参考仓库未修改；其工作区已有未跟踪的 `.DS_Store`，本轮未处理该文件。

```bash
uv run nanomimic play --checkpoint runs/mac-cpu-resume/00000100 \
  --viewer native --device cpu --num-steps-rollout 5 --num-steps 120
uv run nanomimic play --checkpoint runs/mac-cpu-resume/00000100 \
  --viewer viser --device cpu --num-steps-rollout 5 --num-steps 120
uv run pytest tests/test_export.py tests/test_viewer.py
```

### 本轮检查与尚未完成的验收

- T8 阶段 54 项完整测试通过；T9 新增测试后完整 pytest **60 passed
  （31.53 秒）**，ruff、格式检查（42 文件）与 pyright（0 errors）通过。
  第三方警告限于已记录的 torch.jit.script 与旧 ONNX 导出 API 弃用提示。
- 保留五目录组织，只新增规格要求的两个 runtime 实现及对应测试；最大源码
  文件为 `runtime/sim.py`（294 行）。既有夹具和数值误差阈值未修改。
- 无 NVIDIA GPU，未运行 T1、T6 GPU、T7 或最终 sim2sim，未验证显存低于 15GB。
  T8/T9 的实现与 CPU 验证已完成；Git 提交与推送等用户另行授权。
- T1 和 T6 GPU 继续使用下方 T6/T2 记录的命令，状态为
  **待用户在 RTX 5060 Ti 上运行**；默认 num_envs=2048 不变。
- T7：待用户在 RTX 5060 Ti 上运行。准备真实动作文件到下面路径后训练；
  收敛后与参考实现对比成功率和平均跟踪误差，阈值仍须由用户确认。

```bash
uv run nanomimic train --motion-file data/lafan_run1_subject5.npz \
  --device cuda:0 --output runs/t7
```

T8 最终 sim2sim：**待用户在 RTX 5060 Ti 上运行，且须先得到 T7 合格策略**。
下列命令以 T7 默认最终检查点为例，ONNX Runtime 在 CPU 推理，物理环境在 GPU；
从每段动作首帧连续跟踪到末帧，任何 terminated/truncated 都判失败。
只延长验收回合时间以覆盖完整动作，不关闭摔倒终止。此命令本轮未执行，
不以 100 次 CPU 试跑策略代替合格策略。

```bash
uv run nanomimic export --checkpoint runs/t7/01000000 \
  --out outputs/t7.onnx --verify
uv run python - <<'PY'
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from nanomimic.common.specs import SimConfig
from nanomimic.rollout.imagine import actor_goal
from nanomimic.rollout.sensor import HistoryRing, add_noise
from nanomimic.runtime.export import deployment_obs
from nanomimic.runtime.sim import Sim
from nanomimic.training.config import load_config

cfg = load_config(Path("runs/t7/01000000"))
assert cfg.motion_file is not None
session = ort.InferenceSession("outputs/t7.onnx", providers=["CPUExecutionProvider"])
with Sim(SimConfig(cfg.motion_file, num_envs=1, device="cuda:0", seed=cfg.seed)) as sim:
    env = sim._env
    env.cfg.episode_length_s = (sim.motion.num_frames + 1) * sim.robot.dt
    start = 0
    for clip, end in enumerate(sim.motion.clip_ends.tolist()):
        sim.reset()
        sim._command.reset_to_frame(torch.tensor([0], device=sim.device), start)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.sim.sense()
        sim._command.update_relative_body_poses()
        obs = sim._snapshot(env.observation_manager.compute(update_history=True))
        history = HistoryRing(cfg.actor.history_length)
        history.push(add_noise(obs.sensor))
        for frame in range(start, end + 1):
            assert sim.motion_time.item() == frame, "动作时间未连续推进"
            packed = deployment_obs(history.window(), actor_goal(obs.goal)).cpu().numpy()
            action = session.run(["actions"], {
                "obs": packed,
                "time_step": np.array([[frame]], dtype=np.float32),
            })[0]
            obs, _, terminated, truncated = sim.step(torch.as_tensor(action, device=sim.device))
            assert not (terminated | truncated).any(), f"第 {clip} 段第 {frame} 帧失败"
            history.push(add_noise(obs.sensor))
        print(f"第 {clip} 段完整跟踪通过，共 {end - start + 1} 帧")
        start = end + 1
PY
```

## 2026-11-16 · 按职责整理目录（用户确认的方案二）

### 结构与合并

- 用户确认按 `common/`、`models/`、`rollout/`、`training/`、`runtime/` 五个目录
  组织源码；当前实现文件由 19 个减为 16 个，不计 `__init__.py`。
- 原顶层 `specs.py`、`rotation.py`、`networks.py` 移入 `common/`；
  `model.py`、`policy.py` 移入 `models/`；`sensor.py`、`buffer.py`、`imagine.py`
  移入 `rollout/`；`sim.py`、`motion.py` 移入 `runtime/`。
- `goals.py` 与 `rewards.py` 合入 `rollout/imagine.py`，保留目标构造、重锚定、
  奖励与软终止的数值表达式；`sinks.py` 合入 `training/train.py`。
- 其余训练文件移入 `training/`，`training_config.py` 简写为 `config.py`。
  后续 T8、T9 分别使用 `runtime/export.py` 与 `runtime/viewer.py`，本次不提前实现。
- 更新全部源码、测试和工具脚本的导入路径；包入口仍为 `nanomimic.cli:main`，
  CLI 参数和四文件检查点格式不变。各子包初始化只保留说明，不加载子模块。
- 导入规则递归扫描全部子包，对 `common/`、`models/`、`rollout/` 全部模块检查
  mjlab / mujoco / wandb 隔离；mjlab 例外限于完整路径 `runtime/sim.py` 和
  `runtime/viewer.py`，其他目录中的同名文件不获得例外。
- 已同步 `docs/SPEC.md` 的目录、模块职责、任务路径和导入规则，以及 `AGENTS.md`
  与 `README.md`。此前 T6 的「无 SPEC 偏离」未反映目录与规模的差异，表述不够准确；
  本次目录变更已获用户确认，并写入现行规格。

### 验证

- 迁移前：ruff、格式检查、pyright（0 errors）、pytest（47 passed，29.24 秒）通过。
- 迁移后：ruff、格式检查（38 文件）、pyright（0 errors）、pytest
  **47 passed（28.33 秒）** 全部通过；仅有已记录的第三方 torch.jit.script 弃用警告。
- 与迁移前 HEAD 对比：20 个原模块归一化导入路径并移除说明字符串后，全部实现定义
  的语法树一致；现有夹具逐字节一致。最大文件仍为 `runtime/sim.py`（274 行），
  合并后的 `rollout/imagine.py` 为 169 行，`training/train.py` 为 176 行。
- `uv run nanomimic --help` 与 `uv run nanomimic train --help` 正常退出。
- 参考仓库未修改；本次只整理代码组织，GPU 验收仍待用户在 RTX 5060 Ti 上运行。

下方 T0–T6 的文件名、目录和统计数据保留各任务当时的实测记录；现行路径以
`docs/SPEC.md` 第 2 节及上述迁移说明为准。

## 2026-11-16 · T6 训练循环与 CLI（CPU 通过，GPU 待运行）

### 实现与 CPU 实测

- 接入 `nanomimic train`：Collector → WorldModelFitter → PolicyImprover。
  Trainer 只排程；训练次数按 replay_ratio×num_envs/(num_batches×batch×horizon)
  累积，包含预热期的余额。单测验证每步 0.5 次的余额在可用窗口出现后正确消费。
- 世界模型每个窗口预测 horizon 步，因此读取 horizon+1 个状态；默认 horizon=8
  不变。全通道平方误差按 Bryson 权重加权后取均值，范数裁剪为 1.0。
- Actor 对存活概率折扣回报与末端价值反向传播；动力学和 Critic 参数冻结，
  但保留对输入的导数。Critic 使用 lambda=0.95 的 GAE，单测确认实际执行 5×4
  个 minibatch；Actor 和 Critic 均使用 beta2=0.95、范数 1.0 与元素 0.1 裁剪。
- 每个 MLP 的中间 Linear 权重使用发布版 `torch.optim.Muon`，其余参数使用
  Adam；参数分组无遗漏、无重复。学习率默认均为 2e-4，Muon 采用参考的
  `match_rms_adamw` 缩放；SPEC 未规定权重衰减，选择 0，不额外加入正则项。
- 合成监督系统由独立解析积分产生标签，固定种子，小网络、CPU、60 次更新：
  世界模型损失 **0.706855 → 0.00141098**，下降约 99.80%。默认训练宽度与规模未改。
- 真实 mjlab CPU 运行：2 环境、H=3、世界模型和策略 batch=2 / horizon=2，
  先跑 4 次迭代保存，再恢复到第 50 次，累计 25 轮 fit/improve。
  `aot_eager` 与默认 `inductor` 两种编译后端均完成；最终所有网络参数、归一化
  张量均有限。Inductor 最后一次世界模型损失 0.175211、策略损失 -0.493881、
  Critic 损失 0.00105201；这些数字只说明闭环可运行，不是动作跟踪效果验收。
- CPU 编译梯度验收：世界模型和策略的 `aot_eager` 与未编译版本一致
  （atol=1e-6、rtol=1e-5）；策略改进前后动力学 state_dict 严格不变。
- 检查点每 1000 次及结束时保存到独立迭代目录：world_model.pt、actor.pt、
  critic.pt、config.json（schema_version=1）。含归一化、优化器动量、RNG、迭代号、
  更新次数及小数余额。临时目录写完后重命名，已有目录拒绝覆盖。
- `--resume` 先把保存配置交给 tyro 作为默认值，再应用本次显式参数，包括显式
  设回默认值；显式学习率覆盖不会被优化器状态覆盖。恢复逐项验证参数、统计量、
  优化器张量完全一致。仿真和回放重新初始化、预热，不承诺逐步轨迹完全复现。
- 默认日志每 10 次，W&B 仅在显式指定 `--wandb-project` 时创建标量 Sink；
  本机未连接 W&B。已核对 Context7 的 checkpoint、Muon、compile、tyro defaults
  和 W&B scalar API，并核对本地 torch 2.15 的 Muon 实际默认值。
- 最终检查：ruff、格式检查（36 文件）、pyright（0 errors）、完整 pytest
  **47 passed（27.34 秒）**。仅有已记录的第三方 torch.jit.script 弃用警告。
  包内最大文件为 sim.py（274 行）；现有对拍夹具未修改。
- 无 SPEC 偏离。T3、T4、T5 已分别提交；T6 按本次授权独立提交，不推送。

CPU 复现（输出目录已存在时换一个新目录）：

```bash
uv run pytest tests/test_learning.py -s
uv run nanomimic train \
  --motion-file tests/fixtures/motion.npz --device cpu --num-envs 2 \
  --max-iterations 4 --buffer-capacity 16 --replay-ratio 1 \
  --wm.batch 2 --wm.horizon 2 --wm.num-batches 1 \
  --policy.batch 2 --policy.horizon 2 \
  --dynamics.hidden 16 16 16 --actor.hidden 16 12 8 8 --critic.hidden 16 8 \
  --actor.history-length 3 --log-interval 1 --output runs/t6-cpu
uv run nanomimic train --resume runs/t6-cpu/00000004 \
  --max-iterations 50 --log-interval 10 --output runs/t6-cpu-resume
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

CPU 首次 Inductor 编译较慢；可显式加 `--compile-backend aot_eager` 验证编译图，
或 `--no-compile` 运行普通 CPU 调试。受限会话可在 T2 的缓存设置之外设置：

```bash
export TORCHINDUCTOR_CACHE_DIR=/private/tmp/nanomimic-inductor
```

### T1 和 T6 GPU：待用户在 RTX 5060 Ti 上运行

本机是 M1 Pro，无 NVIDIA/CUDA；**未运行 GPU 验收，也未验证显存峰值低于 15GB**。
T1 仍按下方 T2 记录的两档命令测量，默认 num_envs=2048 暂不修改。
T6 请使用默认网络、批量、想象步数和缓冲，只把迭代数设为 50：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv run nanomimic train --motion-file tests/fixtures/motion.npz \
  --device cuda:0 --max-iterations 50 --output runs/t6-gpu
```

在另一个终端从训练开始前采样，训练完成后 Ctrl+C 停止。排除其他 GPU 任务，
检查整张卡显存以涵盖 Warp / MuJoCo 分配；日志中的 PyTorch 峰值不是总显存：

```bash
mkdir -p runs
nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits \
  -lms 200 > runs/t6-gpu-vram-mib.csv
```

训练及显存采样结束后验证检查点与显存记录：

```bash
uv run python - <<'PY'
from pathlib import Path
import torch

directory = Path("runs/t6-gpu/00000050")
for name in ("world_model", "actor", "critic"):
    payload = torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)
    assert payload["iteration"] == 50, "未完成 50 次迭代"
    assert all(torch.isfinite(value).all() for value in payload["model"].values()), name
    if name == "world_model":
        assert payload["progress"]["updates"] > 0, "尚未执行训练更新"
samples = Path("runs/t6-gpu-vram-mib.csv").read_text().splitlines()
peak_gb = max(int(value.strip()) for value in samples if value.strip()) * 1024**2 / 1e9
assert peak_gb < 15, f"显存峰值超限：{peak_gb:.3f} GB"
print(f"50 次迭代检查点有限，总显存采样峰值 {peak_gb:.3f} GB")
PY
```

此合成动作仅用于 T6 稳定性验收；T7 的 lafan_run1_subject5 收敛效果和阈值比较
仍未开展。GPU 数据到手后再补记实际峰值、CUDA 构建与是否通过。

## 2026-11-16 · T5 策略与可微想象

- 实现单任务 Actor（目标编码器、32 维 tanh 潜变量、历史解码器）、特权 Critic、
  目标窗口、重锚定奖励、软终止与逐步 checkpoint 的可微想象。
  数学目标与奖励分别置于 `goals.py`、`rewards.py`，供后续训练与回放复用。
- 对拍最大绝对误差：奖励 2.33e-9、软终止 0、目标窗口 2.99e-7，均低于
  1e-5。额外测试阈值处终止概率 0.5、同姿态的有限梯度及片段末帧钳位。
- 固定种子、4 步想象、H=3、小网络：Actor 所有参数梯度有限，总梯度非零；
  checkpoint 开关的奖励严格一致，梯度满足 atol=1e-6、rtol=1e-5。
  前向和重算均未更改归一化计数，单步没有闭包捕获后续被修改的循环状态。
- 明确测试了计分时序：预测后对当前参考计分，再推进参考时间；末端 Critic
  观测使用推进后的目标。回报测试验证 r_t 乘 s_t、bootstrap 乘 s_H。
- 参考默认 `BaseRunnerCfg.history_length=1`，本项目沿用；编码器与解码器均采用
  2048、1024、512、256 隐藏层（参考 models.py），Critic 为 256、128。
  CPU 测试显式减小网络，不改训练默认值。
- 旋转误差 acos 在 ±1 附近按参考钳位 1e-7；软终止的倾斜误差为投影重力
  z 分量之差；价值仅钳制上界 5×dt/(1−gamma)，允许负值。均与参考数学契约一致。
- 想象从回放中已有的带噪历史开始，后续每步只调用一次统一噪声入口；与参考
  GenCoord 一致，合成 sensor 不额外施加编码器固定偏置，真实采集保留该偏置。
- ruff、格式检查（28 文件）、pyright（0 errors）、pytest（40 passed）通过。
  夹具未修改；本任务无 SPEC 偏离，无 GPU 验收声明。

```bash
uv run pytest tests/test_imagine.py
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## 2026-11-16 · T4 传感器与回放缓冲

- 实现 `synth_proprio`、统一的 `add_noise`、`HistoryRing` 与每环境环形回放。
  合成传感器显式传入默认关节姿态，以保持关节位置为相对值且不依赖全局状态。
- 历史为帧主序、最旧在前；重置行填充同一次噪声采样的整帧。
  小历史使用函数式张量更新，避免原地覆盖想象反向传播仍需读取的数据。
- 槽 t 保存 s_t、带噪 sensor_t、a_t 与该动作产生的 done_t；L 帧窗口仅检查
  前 L−1 条连接无 done，最后一槽可为 done。T6 将采样 horizon+1 帧以取得
  horizon 个合法监督后继，自动重置状态不会被当作旧回合后继。
- `sample_start` 返回包含当前帧的 H 帧历史、对应动作和布尔掩码；False 表示
  首帧填充，True 表示真实数据。窗口不足或已被环形覆盖时，以当前最早可用的
  同回合观测填充并标 False，不读取已经覆盖的历史。起点上一动作从 sensor
  尾部读取，不能把该槽尚未发出的 action 当成上一动作。
- 采样先均衡覆盖有合法窗口的环境，再在各环境合法窗口内均匀抽样；合法性用
  done 累加和计算，历史边界用累积最大值计算，没有逐环境 Python 采样循环。
- CPU 验收覆盖噪声半宽、动作通道无噪声、历史梯度、部分行重置、terminated /
  truncated、环境分层、容量覆盖、H=1、空缓冲与无合法窗口的环境。
- ruff、格式检查（23 文件）、pyright（0 errors）、pytest（34 passed）通过。
  默认缓冲保持每环境 1000 步；现有夹具未改动，无 SPEC 偏离。

```bash
uv run pytest tests/test_sensor_buffer.py
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## 2026-11-16 · T3 可微世界模型

- 实现纯张量表示、半隐式积分、状态重建、FK、损失分组与 Dynamics。
  `rotation.py` 和 `networks.py` 分别承载复用数学与网络构件，保持单文件约 300 行以内。
- fp32 对拍（`atol=1e-5, rtol=0`）：represent、loss_groups、Bryson 权重误差为 0；
  积分 aux 最大误差 1.20e-7、rep 最大误差 2.99e-7；FK 位置/四元数最大误差
  4.18e-7，线速度 2.91e-7，角速度 8.05e-7。现有夹具未改动。
- 零角速度积分及 FK 的梯度有限且非零；动作到动力学输出的梯度通过。
  RunningNorm 分批合并统计量与一次性总体统计一致，前向不隐式更新。
- 未明确的归一化细节采用合并批次矩，标准差下限 0.01；只由采集阶段显式更新，
  避免想象、checkpoint 重算和评估污染统计。均值、方差、计数均进 state_dict。
- 积分仍采用规格规定的完整 dt（substep 系数 1.0）。FK 独立按运动树传播位姿和
  body 原点扭量，包含铰链原点偏移；没有引入参考类层次或仿真器依赖。
- 默认 Dynamics 保持 3 × 2048；CPU 梯度测试显式使用小网络。
- 检查通过：ruff、格式检查（20 文件）、pyright（0 errors）、pytest（27 passed）。
  第三方 torch.jit.script 弃用警告与 T2 相同。无 GPU 验收声明，无 SPEC 偏离。
- 本次用户已授权 T3–T6 各自独立提交、不推送；T1 保持待用户在 RTX 5060 Ti 上运行。

```bash
uv run pytest tests/test_model.py
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## 2026-11-16 · T2 仿真适配与对拍夹具

### 状态与实测

- 按用户本次要求，先做 M1 Pro 上可以验收的任务。T0 已有提交；T1 仍待 GPU
  实测，本次推进 T2，不把 T1 记为完成，默认 `num_envs=2048` 暂不调整。
- 实现 `Sim`、`Obs`、`RobotSpec`、`Motion`、`ContextSpec`，注册
  `Nanomimic-Tracking-Flat-Unitree-G1`。配置为 frozen dataclass，
  `tools/check_sim.py` 用 tyro 暴露 CPU/CUDA 验收参数。
- 使用现有只读参考仓库 `../supermimic`，提交
  `abc124c1bd5976341034668af3d12c087f2f1f55`；没有修改参考仓库，也没有安装其私有 mjlab fork。
- 环境：Python 3.13.7、torch 2.15.0、mjlab 1.6.1、MuJoCo 3.11.0、Warp 1.19.0；
  设备明确为 `cpu`，CUDA 不可用。
- 独立验收命令跑通 **2 个环境 × 100 步随机动作**，共 3 次自动重置。
  首次运行的步进部分（含契约检查）耗时 3.17 秒；此值不用于 GPU 性能推断。
- 每步检查：状态 `[2, 71]`、目标 `[2, 670]`、传感器 `[2, 93]`、
  上下文 `[2, 33]`，所有值有限，上下文在 `[-1, 1]` 内。
- 机器人：29 关节、30 个运动树 body、14 个跟踪 body，锚点 `pelvis`，
  控制步长 0.02 秒，软终止阈值分别为 0.25 与 0.8。
- 最终检查：`ruff check`、`ruff format --check`（15 个文件）、`pyright`
  （0 errors）与 `pytest`（23 passed，6.60 秒）通过。唯一警告来自 mjlab 导入时的
  `torch.jit.script` 弃用提示，与 T0 已记录的问题相同。

### 契约与决策

1. **发布版接口**：通过 Context7 文档和本地 1.6.1 源码核对，重置模式名称确为
   `adaptive`。发布版在整段动作结束时重采样、写入机器人状态；本项目在 `sim.py`
   内用薄子类将推进改为片段末帧钳位，保留其 adaptive 采样与失败率更新。
2. **片段端点**：NPZ 可选 `motion_ends` 沿用参考的右开端点；读入后转换成
   `Motion.clip_ends` 的包含式末帧索引。未提供时是一段动作。文件必须包含完整
   30 个机器人 body 的数据，之后选取 14 个跟踪 body；不接受缺字段、错误形状、
   非有限值、非单位四元数或错误帧率。若存在 `fps`，必须为 50，不自动重采样。
3. **传感器**：沿用参考的部署含义：关节位置是相对默认姿态的编码器读数，包含
   固定的编码器偏置；基座角速度来自 IMU。这里的“干净”指未添加逐帧均匀噪声，
   不移除域随机化偏置。噪声入口留给 T4 的 `sensor.py`。
4. **上下文**：三个通道实际为 `1 + 3 + 29 = 33` 维，顺序是脚摩擦、质心偏移、
   编码器偏置。摩擦读取实际全局 geom 索引；质心先减名义 `body_ipos`，再按
   配置中的偏移范围归一化。参考实现直接归一化绝对 `body_ipos`，可能超出
   `[-1, 1]`；本项目按 SPEC 的范围要求修正，不以 clamp 隐藏越界。
5. **自动重置**：沿用 mjlab 的自动重置语义。`step()` 的 done 标志描述旧回合，
   done 行返回的是新回合观测，上一动作清零；后续缓冲不能把它当成旧回合的后继。
   返回观测、奖励、done 和动作时间均为快照，不会被下一步原地覆盖。
6. **解耦**：dataclass 放在纯张量 `specs.py`，NPZ 校验放在 `motion.py`；
   仿真器导入仍仅在 `sim.py`。FK 常量使用完整树的父节点、固定变换、关节轴、
   铰链原点与 qpos0，后续 T3 不需要导入 mjlab 或 MuJoCo。
7. **类型检查**：将工具脚本也纳入 pyright。MuJoCo 的公开 C 扩展动态导出缺失
   可解析类型桩，仅在工具中对应的单行导入精确抑制 `reportAttributeAccessIssue`；
   项目其他代码继续执行原有检查规则。

### 夹具与验证范围

- `tools/make_fixtures.py` 调用参考实现生成表示、积分、损失分组、Bryson 权重、
  目标窗口、奖励和软终止真值；没有把参考代码复制进本项目。
- 积分显式采用 SPEC 的 `substep_coeff=1.0`；参考工厂在 decimation=4 时采用
  0.625。差异是执行已有规格，不是新增选择。
- 发布版 mjlab 不含参考私有 fork 的 `ForwardKinematics`。FK 夹具由 MuJoCo
  `mj_forward` 和 body 原点处的 `mj_jac` 独立计算，速度为世界系；不把质心速度
  误当成 link 原点速度。夹具同时保存完整运动树常量供 T3 对拍。
- 合成动作共 96 帧、两个片段，只用于接口验收；不代表 `lafan_run1_subject5`
  的训练效果。`manifest.json` 记录种子、依赖版本、参考提交、源码及产物 SHA-256。
  所有夹具均通过生成器生成，未手工编辑。
- 在独立临时目录再次运行生成器，两个 NPZ 与 `manifest.json` 均逐字节一致。
- 测试覆盖随机步进、自动重置、超时、注册配置隔离、返回值快照、实际随机化参数、
  运动树常量、片段边界和错误输入。状态与目标观测对参考夹具使用 `atol=1e-5, rtol=0`。
  mjlab 的 `get_observations()` 会返回缓存；测试写入状态后显式刷新观测管理器。
- T3/T5 的数值实现与梯度验收尚未开始；当前只是为它们生成真值。
- 用户已于 2026-11-16 明确授权提交 T2；提交前复跑 ruff、格式检查、pyright 与
  pytest（23 passed）全部通过。

复现命令（普通终端）：

```bash
uv run python tools/make_fixtures.py --reference ../supermimic
uv run python tools/check_sim.py
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

受限 agent 会话中为上述命令设置可写缓存目录；普通终端不需要：

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
export MPLCONFIGDIR=/private/tmp/nanomimic-mpl
export XDG_CACHE_HOME=/private/tmp/nanomimic-cache
export WARP_CACHE_PATH=/private/tmp/nanomimic-warp
```

### T1：待用户在 RTX 5060 Ti 上运行

先核对 CUDA 构建，再分别运行两档环境数。下面使用同一份本地合成动作，输出的是
含形状/有限性检查的环境步数/秒；不能和去掉检查的纯物理吞吐直接比较。
另一个终端运行 `nvidia-smi` 采样显存，排除其他 GPU 进程后记录峰值。
拿到两档实测数据前不确定最终默认环境数。

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi --query-gpu=timestamp,name,memory.used --format=csv -lms 200
```

```bash
uv run python tools/check_sim.py --device cuda:0 --num-envs 2048 --steps 1000
uv run python tools/check_sim.py --device cuda:0 --num-envs 4096 --steps 1000
```

GPU 命令本机未运行。T6、T7、T8 的 GPU 验收留到对应任务，不在 T2 宣称通过。

## 2026-11-15 · T0 脚手架

### 目录与环境

- 开发文档移到 `docs/`（`docs/SPEC.md` 为规格，本文件为记录），`AGENTS.md` 留在根目录
  供 agent 自动加载，路径引用已同步；移动后复跑全套检查通过。
- 开发机 macOS Darwin 27.2.0 / Apple Silicon（arm64），无 NVIDIA GPU；
  uv 0.13.8、git 2.54.0、Python 3.13.7（uv 管理）。
- `uv sync` 解析 154 个包、安装 133 个。锁定版本：torch 2.15.0、mjlab 1.6.1、
  tyro 1.0.17、onnx 1.23.2、onnxruntime 1.32.0、pytest 9.2.0、ruff 0.17.5、pyright 1.1.414。
- `ruff check`、`ruff format --check`（7 个文件）、`pyright`（0 errors）、
  `pytest`（3 passed）、`nanomimic --help` 全部通过。
- 本机 torch 为 macOS 构建：`torch.version.cuda is None`、`torch.cuda.is_available() is False`，符合预期。
- agent 会话受文件沙箱限制，uv 缓存放在工作区 `.uv-cache/`（已 gitignore，约 1.4 GB，
  可随时删除）；正常终端环境无需此设置。

### 决策与理由

1. **Python 3.13，不升 3.14**（2026-11-16 复核）。
   - mjlab 1.6.1 的 PyPI 元数据为 `Requires-Python: >=3.10,<3.14`，0.0.5 起所有版本都带上限。
   - uv 解析时只取 `requires-python` 下界、忽略上界（uv 文档 `pip/compatibility.md`
     明确说明），所以 3.14 下 `uv sync` 照样成功；但 `pip install mjlab==1.6.1` 会直接
     报 `No matching distribution found`。**「uv sync 能过」不能作为兼容性判据。**
   - 3.14 下功能实测均正常（G1 跟踪任务 reset + 20 步、cartpole、warp CPU 内核、
     ONNX 导出与推理误差 1.2e-7、`torch.compile`）。唯一实质风险是 `torch.jit.script`
     在 3.14 被 torch 标为 `not supported ... may break`，而 mjlab 的
     `utils/lab_api/math.py` 有 28 个 `@torch.jit.script` 函数；3.13 下只发 deprecated 级警告。
   - 结论：退回 3.13（`docs/SPEC.md` 第 1 节允许）。升级前提是 mjlab 放开上界
     **且** torch 在 3.14 上仍支持 `jit.script`，缺一不升。
2. **mjlab 放在默认依赖**：`uv sync` 一次装好仿真依赖，T1/T2 可直接使用；
   `wandb`、`tensordict` 只是 mjlab 的传递依赖，本项目代码不直接导入。
3. **CLI 在 T0 只搭骨架**：三个子命令与参数（`--checkpoint`、`--viewer`、`--out`、
   `--verify`）已可用，执行逻辑留给 T6、T8、T9，当前以退出码 1 提示「尚未实现」。
   命令行契约先固定，后续只填实现。
4. **导入规则测试按 `docs/SPEC.md` 分层**：`tests/test_import_rules.py` 用 AST 检查
   model、policy、sensor、imagine、buffer 不得导入 mjlab、mujoco、wandb；全包范围只有
   sim.py、viewer.py 可以导入 mjlab。核心文件由后续任务创建，出现后自动纳入检查。
5. **ruff 关闭 RUF001/002/003**：文档、注释和输出按 AGENTS.md 使用中文全角标点，
   这三条同形字歧义规则会误报；其余 E/F/I/UP/B/SIM/RUF 规则保持开启。
6. **pyright 用 standard 模式**：`docs/SPEC.md` 只要求无报错，standard 已能覆盖本项目的
   类型错误，strict 会被第三方类型桩的细节淹没；后续出现关键数值代码问题时再局部提高。
7. **默认索引写进 `pyproject.toml`**：`[[tool.uv.index]]` 里声明清华镜像并设
   `default = true`，锁文件因此不依赖某台机器的全局 uv 配置（已在干净 `HOME` 下验证）。
   注意 `uv lock --no-config` 会忽略 pyproject 里的索引配置，不能用来验证镜像设置。

### 与参考仓库的差异

T0 未使用参考仓库：`../supermimic-ref` 尚不存在，脚手架不需要其中的常量或契约。
T2 开始按 `docs/SPEC.md` 克隆为只读参考，只取常量、夹具和 mjlab 契约，不复制其文件结构或代码。

### 待用户在 RTX 5060 Ti 上运行

T0 无 GPU 验收项，但 torch 的 CUDA 构建需在 GPU 机上核对（本机为 macOS 构建）：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

期望 `torch.version.cuda` 不低于 12.8 且 `torch.cuda.is_available()` 为 `True`。
若默认 wheel 不满足，按 PyTorch 2.15 官方说明换 CUDA 12.8+ 构建并把实际版本补记到本节。

### 待用户确认

- T0 无偏离 `docs/SPEC.md` 的决定需要确认。
- 第 6 节第 2 条的效果阈值要等 T7 与参考实现对比后再提交确认。

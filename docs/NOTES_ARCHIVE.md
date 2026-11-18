# 决策与实测归档

本页按主题保存已归档的决策、实测结果和待办。项目约定见[开发规格](SPEC.md)，新决定写入[决策记录](NOTES.md)。

2026-11-18 首次归档。以下训练数据和运行状态均来自当时的记录。

## 当前状态与待办

| 任务 | 最近结论 |
|---|---|
| T0–T6 | 已通过，包括 2026-11-16 的 T1、T6 RTX 5060 Ti 实测；T6 GPU 使用当时的 2048 环境配置 |
| T7 | 已启动真实动作训练；已记录的 5000 次迭代评估为 53/64 成功，效果尚未验收 |
| T8 | 实现与 CPU 数值验收完成；完整动作闭环且不摔倒的 sim2sim 待 T7 合格策略 |
| T9 | 实现、Mac CPU 回放及虚影对齐实测完成；试跑策略不代表训练效果 |

尚未完成的事项：

- **效果阈值待用户确认**：没有可用的同段动作 SuperMimic 参考检查点或评测结果；
  须按同一协议比较完整成功率和平均误差，再提交阈值，不宣称已达到参考水平。
- **默认规模确认保留**：2026-11-17 已将 `num_envs=4096`、`max_iterations=100000`
  写入配置、SPEC 与 README；原记录注明需用户确认，本次整理不将其改为已确认。
- **待用户在 RTX 5060 Ti 上运行**：4096 环境完整训练峰值低于 15 GB 的验收，
  新布局下第 1000 轮保存与第 1001 轮结束补存，以及 T7 后的 T8 最终 sim2sim。
  2048 环境的 T6 数据不能代替新默认规模验收；命令见文末。

## 环境与基础选择（2026-11-15–16）

| 环境 | 已记录版本与设备 |
|---|---|
| 初期 CPU 开发 | macOS Darwin 27.2.0、M1 Pro / arm64；Python 3.13.7、torch 2.15.0，无 CUDA |
| T1 / T6 GPU | Linux x86_64、i7-14700KF、RTX 5060 Ti 16GB（16311 MiB、sm_120）；Python 3.13.15、torch 2.15.0+cu132、PyTorch CUDA 13.2 |
| GPU 驱动与仿真栈 | 驱动 615.71.09、CUDA UMD 13.4；mjlab 1.6.1、MuJoCo / MuJoCo Warp 3.11.0、Warp 1.19.0（启动日志 Toolkit / Driver 13.4） |
| 初期工具与导出栈 | uv 0.13.8、tyro 1.0.17、onnx 1.23.2、onnxruntime 1.32.0、pytest 9.2.0、ruff 0.17.5、pyright 1.1.414 |

- **Python 退回 3.13**：mjlab 1.6.1 声明 `>=3.10,<3.14`；uv 解析忽略该上界，
  因而 `uv sync` 成功不能证明支持 3.14。3.14 功能试跑虽正常（含 ONNX 误差 1.2e-7），
  但 torch 将其 `jit.script` 标为不支持，mjlab 仍有 28 个相关装饰函数。
  升级须同时满足 mjlab 放开上界、torch 支持该路径；退回符合 SPEC 第 1 节。
- mjlab 放入默认依赖；其传递依赖 `tensordict` 不在项目代码中使用，W&B 只作可选标量 Sink。
  uv 默认清华索引写入 `pyproject.toml`，已在干净配置下验证，避免依赖机器全局设置。
- ruff 因中文全角标点关闭 RUF001/002/003；pyright 使用 standard，工具脚本同样纳入检查。
  MuJoCo 动态导出缺少类型桩时，仅对工具内对应单行导入精确抑制类型错误。
- **2026-11-16 用户确认五目录组织**：目标与奖励合入 `rollout/imagine.py`，
  标量日志合入 `training/train.py`，已同步 SPEC、AGENTS 与 README。
  迁移前后均 47 项测试通过；20 个原模块在归一化导入并移除说明字符串后 AST 一致，
  夹具逐字节一致。导入规则递归扫描，mjlab 例外仅为 `runtime/sim.py`、`runtime/viewer.py`。

## 仿真、数学与训练契约（2026-11-16）

### T2：仿真适配与夹具

- 只读参考为 `../supermimic`，提交 `abc124c1bd5976341034668af3d12c087f2f1f55`；
  只取常量、生成真值、核对契约，没有复制其架构或使用私有 mjlab fork。
- CPU 随机动作 **2 环境 × 100 步**通过，3 次自动重置，步进含检查耗时 3.17 秒。
  状态 / 目标 / sensor / ctx 分别为 `[2,71]`、`[2,670]`、`[2,93]`、`[2,33]`；
  值有限，ctx 在 `[-1,1]`。机器人为 29 关节、30 树 body、14 跟踪 body，
  锚点 `pelvis`，dt=0.02 秒，软终止阈值 0.25 / 0.8。
- 发布版采样模式确为 `adaptive`；薄子类只把动作推进改成片段末帧钳位，保留采样与失败率更新。
  NPZ 的可选 `motion_ends` 是右开端点，读入转为包含式 `clip_ends`；缺省视为单片段。
  文件须含完整 30 body，严格校验字段、形状、有限性、单位四元数及 50 Hz 帧率，不自动重采样。
- “干净” sensor 仍含域随机化编码器偏置，关节位置相对默认姿态；逐帧噪声统一另加。
  ctx 顺序为脚摩擦 1、质心偏移 3、编码器偏置 29。质心先减名义 `body_ipos` 再归一化；
  参考直接归一化绝对位置可能越界，本项目按 SPEC 修正，不用 clamp 掩盖问题。
- `step()` 的 done 描述旧回合，done 行观测属于自动重置后的新回合，上一动作清零；
  返回值均为快照。缓冲不将新回合观测当成旧回合后继；写状态后测试显式刷新观测缓存。
- 夹具由 `tools/make_fixtures.py` 生成，96 帧、两个合成片段，仅作接口与数值验收。
  积分按 SPEC 使用 `substep_coeff=1.0`，参考 decimation=4 的工厂默认 0.625。
  发布版缺少私有 FK，故以 MuJoCo `mj_forward` / `mj_jac` 独立生成 body 原点世界系速度真值。
  manifest 含种子、版本、参考提交与源码 / 产物 SHA-256；独立重生成全部逐字节一致，未手改夹具。

### T3–T5：数值、历史与可微想象

fp32 对拍最大绝对误差如下，统一验收 `atol=1e-5, rtol=0`：

| 项目 | 最大绝对误差 |
|---|---|
| represent、loss_groups、Bryson 权重 | 0 |
| 积分 aux / rep | 1.20e-7 / 2.99e-7 |
| FK 位置与四元数 / 线速度 / 角速度 | 4.18e-7 / 2.91e-7 / 8.05e-7 |
| 奖励 / 软终止 / 目标窗口 | 2.33e-9 / 0 / 2.99e-7 |

- RunningNorm 用合并批次矩、标准差下限 0.01；只由采集阶段显式更新，前向、评估与
  checkpoint 重算均不更新，均值 / 方差 / 计数入 `state_dict`。零角速度积分、FK 与动作梯度均通过。
- HistoryRing 为帧主序、最旧在前；重置行用同一次噪声填满历史，函数式更新保留梯度。
  回放槽 t 存 `s_t、sensor_t、a_t、done_t`；L 帧窗口只检查前 L−1 条连接无 done。
  起点历史包含当前帧，缺失或已覆盖部分以同回合最早可用帧填充并标 False；
  上一动作取 sensor 尾部。采样按环境分层，合法性与历史边界向量化计算。
- 沿用参考 H=1、32 维 tanh 潜变量；Actor 编码器与解码器均为 2048/1024/512/256，
  Critic 为 256/128。CPU 测试使用小网络，不改变默认网络。
- 预测后对当前参考计分，再推进动作时间；末端 Critic 用推进后的目标。
  旋转误差 acos 在 ±1 附近钳位 1e-7；倾斜误差为投影重力 z 分量差；
  价值仅钳上界 `5×dt/(1−gamma)`，允许负值；回报的奖励乘 `s_t`，bootstrap 乘 `s_H`。
- 想象从已有带噪历史开始，后续每步只加一次统一 sensor 噪声，不重复施加编码器固定偏置。
  4 步、H=3 的 Actor 梯度有限且非零；checkpoint 开关奖励严格相同，梯度满足
  `atol=1e-6, rtol=1e-5`，不污染归一化计数。

### T6：训练与恢复的 CPU 验证

- Collector → WorldModelFitter → PolicyImprover；训练额度按
  `replay_ratio×num_envs/(num_batches×batch×horizon)` 累积，含预热期余额。
  世界模型取 horizon+1 个状态预测 horizon 步；Actor 更新冻结动力学和 Critic 参数，保留输入导数。
- 中间 Linear 权重用 Muon，其他参数用 Adam，分组无重复或遗漏；
  Muon 沿用 `match_rms_adamw`，规格未定权重衰减，选择 0，避免额外正则项。
  Critic 的 GAE lambda=0.95，实际执行 5×4 个 minibatch；梯度裁剪沿用 SPEC。
- 独立解析标签、固定种子、小网络 CPU 60 次更新：世界模型损失
  **0.706855 → 0.00141098（下降 99.80%）**。
  真实 mjlab 2 环境、H=3、batch/horizon=2，从 4 次恢复到 50 次、累计 25 轮 fit/improve；
  `aot_eager` 与 Inductor 均完成，参数与归一化有限。Inductor 最终 WM/Actor/Critic 损失为
  0.175211 / -0.493881 / 0.00105201，只证明流程可运行。
- 编译 / 未编译梯度满足 `atol=1e-6, rtol=1e-5`，策略更新前后动力学状态严格不变。
  检查点四文件保存归一化、优化器、RNG、迭代号和排程进度，临时目录完成后重命名，拒绝覆盖。
- 续训先读保存配置，再应用显式 CLI 参数；显式学习率覆盖不被优化器状态覆盖。
  参数、统计量与优化器张量恢复一致；仿真和回放重新初始化、预热，不承诺逐步轨迹复现。

## GPU 实测与编译边界（2026-11-16）

T1 / T6 均用上述 RTX 5060 Ti 环境；每 200 ms 采样整卡显存，包含桌面、PyTorch、
Warp、MuJoCo，未扣基线；采样峰值不保证捕获更短瞬时峰值。原始产物不入 Git。

### T1：两档仿真规模

同一份 96 帧合成动作、seed=0、每档独立进程。构造与 reset 后开始计时，首尾同步 CUDA，
含随机动作、契约检查与重置，无额外预热，不等同纯物理或完整训练吞吐。

| 环境数 | 控制步数 | 耗时 | 环境步数/秒 | 重置次数 | 整卡峰值 MiB / GiB |
|---|---|---|---|---|---|
| 2048 | 1000 | 17.84 秒 | 114795.9 | 48238 | 948 / 0.926 |
| 4096 | 1000 | 28.81 秒 | 142176.8 | 96324 | 1247 / 1.218 |

两档退出码均为 0，无 CUDA 或接触容量错误。基线分别为 404–405 / 405–406 MiB，
采样 381 / 608 点；2048 首次触发 Warp 编译，4096 使用缓存，未重复取最好值。
4096 吞吐高 23.85%、峰值多 299 MiB；当时为完整训练留余量保留 2048，后续默认值见数据节。
环境、日志、CSV 与汇总在 `runs/t1-gpu/`；动作夹具 SHA-256 为
`c4030977382668861961dddc3879404d7bbc448417d2dae4b3d94d9b963f988d`。

### T6：2048 环境、50 次迭代

默认网络、batch、horizon、1000 步缓冲、replay_ratio=16、逐步 checkpoint、Inductor、
fp32/TF32，seed=0，使用合成夹具；只覆盖动作、迭代数、输出路径。退出码 0，完成 10 轮
fit/improve、排程余额 0；WM/Actor/Critic 的 Adam 更新计数为 100/10/200。
三个检查点含归一化的 74 个模型张量、162 个优化器张量全部有限，无 NaN 或梯度异常。

| 第 50 次指标 | 值 |
|---|---|
| WM / Actor / Critic 损失 | 0.136690 / -1.41326 / 0.0411164 |
| 仿真奖励 / 重置比例 | 0.0605439 / 0.019043 |
| PyTorch 已分配 / 保留峰值 | 2.98491 / 3.43094 GB |
| 整卡采样峰值 | **4610 MiB = 4.502 GiB = 4.833935 GB < 15 GB** |

1165 个整卡样本覆盖启动前至退出后，前后基线 618 / 689 MiB，已核实无其他训练任务。
北京时间 22:36:41–22:39:40，耗时 178.42 秒含初始化、编译、训练与保存，使用已有缓存，
不代表冷启动或稳定吞吐。检查点现位于 `runs/t6-gpu/checkpoints/00000050/`，日志与核验在
`runs/t6-gpu-observation/`，显存原始数据为 `runs/t6-gpu-vram-mib.csv`。
本项验收稳定性与当时规模的资源，不代表收敛效果。

### 编译范围的选择与证据

- 原整段 32 步编译约 30 分钟未输出第 10 次指标，RSS 约 18 GiB，最终 SIGTERM、退出 143，
  无检查点，不计验收。改为编译 `imagine_step`，Python 循环复用，每步保留 checkpoint、
  RNG 重算及跨步梯度；其他 rollout / 损失仍编译，未靠关闭编译通过验收。
- 撤回 `nested_compile_region`：2 步对照通过，4 步梯度最大差 **0.003919743**，超过阈值。
  最终普通 `torch.compile` 的 2/4 步对照均满足原梯度阈值；未确诊为框架正确性 BUG。
- 原产物的想象前向模块为 9,801,891 字节、96,333 行、338 个 Triton kernel、6681 个调用点；
  已有 Actor 损失前后向模块，未找到整段想象反向模块。CPU B=2、只统计 FX 图的诊断显示
  1/4/32 步为 **4224/16149/127449 节点**，每档一个区域，证实循环展开成大图。
  诊断在 `runs/t6-compile-audit/`，不替代 GPU 编译计时。
- 最终方案 Dynamo 编译汇总 119.3145 秒，代码生成 / 融合调度 / 自调优为
  61.8799 / 23.5295 / 35.6922 秒，指标嵌套不能相加。
  证据支持缩小编译范围；原路径最慢 pass、是否存在 Inductor 缺陷、同空缓存速度比仍未确认。
  不能把 178 秒与原中断时间当受控提速比。选择依据为
  [PyTorch 区域编译文档](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)。

## T7：真实动作训练与评估（2026-11-16）

- 动作为 `lafan_run1_subject5`，11890 帧、50 Hz、237.8 秒、29 关节、完整 30 body；
  SHA-256：`5e414d9800a0ffce26bd9667dda65ac6c711f287df77c4a5d105c32a9e1d7619`。
  迁移后原文件保存在 `data/processed/lafan1-g1-v1/legacy/lafan_run1_subject5.npz`。
- 真实动作 GPU 预检 2048 环境 × 100 步通过，重置 7314 次。北京时间 23:14:55、
  代码 `17884a9` 从头启动，未恢复 T6；当时为 2048 环境、上限 1000000，其余默认，
  每 1000 次保存。只读参考的上限同为 1000000、环境默认 4096
  （`runners/supermimic_runner.py:72`、`scripts/train.py:28`），不代表实际收敛次数。
- 检查点与评估现位于 `runs/t7-lafan/{checkpoints,evaluations}/`；启动命令、动作哈希、
  进程信息、日志和显存采样在 `runs/t7-lafan-observation/`。
  当时另有每 5000 次及最终检查点评估的监测进程，信息见 `evaluation-watcher.json`、`status.json`。

### 评估协议

`tools/evaluate_tracking.py` 只接受单片段；默认 seed=0、64 次并行首次尝试：

- 从第 0 帧到末帧，11890 帧对应 **11889 个控制步**；关闭动作探索和初态扰动，
  保留三项域随机化与 sensor 噪声，冻结网络与归一化，独立进程不改变训练 RNG。
- 延长原 10 秒超时至动作结束之后，关闭自动重置；真实失败终态计入误差，
  失败后局部重置且不再计分，提前超时算失败。reset 的命令刷新会推进到帧 1，
  因此显式恢复帧 0，并逐步断言时间连续，拒绝跳帧或短回合被判成功。
- step 后重算真实 body 的重锚定指标，不读旧缓存。body 位置 / 朝向为 14 个跟踪 body
  的距离均值（m）/ 角度均值（rad）；joint 为全关节向量 L2 范数。
  报告成功率、存活时间、完成比例、逐尝试结果、尝试期间加权误差与仅成功尝试误差，
  无成功时后者为 null。参考策略必须用同协议再比较阈值。
- API 依据为 mjlab 1.6.1 的 `tasks/tracking/mdp/commands.py:215,320` 与
  `envs/manager_based_rl_env.py:358,469`；评估配置不作用于训练实例。

### 已记录的阶段结果

| 迭代 | 完整成功 | 平均存活秒 | 平均完成 | body 位置 m | body 朝向 rad | 锚点世界位置 m |
|---|---|---|---|---|---|---|
| 1000 | 0/64 | 6.5984375 | 2.7750% | 0.1443023 | 0.7133235 | 1.5285354 |
| 2000 | 0/64 | 38.44875 | 16.1699% | 0.1129856 | 0.5261699 | 4.5113266 |
| 5000 | 53/64（82.8125%） | 213.334375 | 89.7192% | 0.0413590 | 0.1708915 | 3.0760728 |

1000/2000 次均为任务终止、无提前超时，存活范围为 2.16–30.28 / 2.04–104.62 秒。
5000 次仅成功尝试的 body 位置误差为 0.0408078 m，仍有 11 次失败，未达到参考水平验收。
表中误差覆盖各自存活区间，不能只比较低误差而忽略完成比例与锚点漂移。
原始 JSON 为 `runs/t7-lafan/evaluations/eval-00001000-seed0.json` 等。

第 1000 次检查点已核验 200 轮更新、余额 0、74 个模型张量有限。
截至当日 23:31:52 的 5078 个整卡样本峰值 **5535 MiB = 5.80386816 GB**（含桌面与评估），
这是阶段观测，非整次训练最终峰值。无并行评估的 23:31:16–23:41:06 窗口共 2950 点，
GPU 利用率均值 86.9444%、中位数 96%；不能据剩余显存直接推断扩容收益。
`tools/plot_training.py` 已用历史日志、评估及显存 CSV 生成并目视检查
`outputs/t7-curves/` 的训练与资源 PNG/SVG；缺失损失不补零，21 点向后均值只用于显示。

## 动作数据与默认规模（2026-11-17）

- `lvhaidong/LAFAN1_Retargeting_Dataset` 的 G1 数据实际为 **40 段** CSV（非原版 77 段），
  已全部下载，以 mjlab `csv_to_npz` 从 30 Hz 重采样到 50 Hz 并计算 FK。
  逐段约 755 MB；合并 **441120 帧、40 个片段、约 754 MB**，通过形状、有限性、
  单位四元数与帧率校验。顺序为 `LAFAN1_CLIPS` 字典序，端点按帧数累加。
- 多片段 CPU 2 环境、3 次迭代通过，采样可进入任意片段，目标窗口不跨片段。
  合并数据超过 ONNX 15000 帧上限，导出须使用单段或裁剪；新目录见下一节。
- 转换的局部兼容：`fallAndGetUp2_subject2` 的 body 角速度自检 fp32 最大差
  **1.2815e-5**，CPU/CUDA 均复现。只在转换调用内包装第三方 `assert_close`：
  先用原参数断言，仅失败且绝对差 **<1e-3** 才放过并打印差异，结束立即恢复；
  更大误差仍报错，不改输出数据、mjlab 源码或项目数值验收阈值。
- 新默认值为 **4096 环境、100000 次上限**，其余训练超参不变，确认状态见开头。
  每环境 1000 步回放容量估算由 1.7433 增至 3.4866 GiB，尚非完整训练 GPU 实测。
  每采集迭代的训练额度从 0.2 增至 0.4，同迭代数的样本量与训练工作翻倍；
  **100000×4096 的训练量约为旧 1000000×2048 的 20%**（原记录 2% 为算术笔误）。
  上限不代表收敛，T1 仿真吞吐增幅也不代表完整训练提速。

## 运行产物、帮助与迁移（2026-11-17–18）

### 日志与目录

- 用户授权终端默认每 100 次打印、每 1000 次保存检查点与曲线；原始 `metrics.jsonl`
  每轮写完整精度标量，无训练更新时不伪造损失。曲线 PNG/SVG 用 Agg，无需窗口，
  21 点向后均值；WM/Critic 损失全为正时用对数轴，图片临时写完再替换。
  Matplotlib 3.11.2 从传递依赖改为直接依赖 `>=3.11,<4`，未升级版本。
- `--output` 为根目录，每次创建北京时间时间戳子目录，同秒加序号；续训也创建新目录，
  保存配置保留输出根目录，本次指标不拼接历史。训练前保存 `config.json`、`run.json`
  （schema、创建时间、续训来源），完整检查点保存后原子更新 latest 索引，正常结束补存并刷新曲线。
- CPU 2 环境、5 迭代、日志/保存周期=2：终端仅 2/4 轮，原始指标完整 1–5，
  检查点与曲线更新于 2/4/5；续训仅记录 5–8 轮，省略 output 可继承根目录。

| 产物 | 现行路径 |
|---|---|
| 原始 / 逐段 / 合并动作 | `data/raw/lafan1/g1/` / `data/processed/lafan1-g1-v1/clips/` / `data/cache/lafan1-g1-v1/all.npz` |
| 单次训练 | `runs/<根目录>/YYYYMMDD_HHMMSS/`；根目录保存配置、运行信息、指标与曲线 |
| 检查点 | 运行目录的 `checkpoints/迭代号/`，四文件契约；`checkpoints/index.json` 指向 latest |
| 默认评估 / ONNX | 运行目录的 `evaluations/` / `exports/actor-迭代号.onnx`；显式输出参数优先 |

### 已执行迁移（2026-11-18 00:28:36，北京时间）

- `tools/migrate_layout.py` 默认只展示计划，`--apply` 执行；预检全部冲突、拒绝覆盖，
  先存计划、rename 后逐文件核对设备号/inode/大小，不重写数值或历史配置。
- 完成 **343 项移动、1132 个文件**：40 CSV、40 逐段 NPZ、合并与早期单段 NPZ、
  250 份四文件检查点、50 份评估 JSON。**82/82 数据文件 SHA-256 一致**，
  迁移后仍有 250 份检查点，再次计划为 0 项。记录：`data/migrations/20261118_002836_533345.json`。
- 旧 `data/lafan`、`data/lafan1_all.npz`、`data/lafan_run1_subject5.npz` 保留相对软链接；
  早期单段置于 `legacy/`，不以后来同名转换数据替换，确保旧检查点使用原内容。
  `runs/t6-gpu/`、`runs/t7-lafan/` 保留名称，内部迁移；`runs/g1/` 的四份平铺检查点
  收进 `legacy/`，保留 `20261118_001006/` 指标。历史根配置原样复制最新检查点配置，
  历史 `run.json` 仅记迁移时间，不推测启动时间或补造配置。
- `runtime/paths.py` 统一定位与 JSON 原子写入，旧平铺检查点参数仍兼容。
  `runs/t7-lafan/00245000` 已实测解析并 CPU 加载 245000 次 Actor，旧动作路径指向原文件；
  `make_motion.py --list` 显示 40 段 CSV/NPZ 就绪。CPU 训练/续训、评估、ONNX 默认目录均验过。
- 本阶段只整理目录，保留全部检查点；保留策略、最佳检查点、完整数据 manifest、
  环境指纹与逐段评估留待后续，不在此次扩大范围。

### CLI 中文帮助

- tyro 1.0.17 内部按字符数计宽导致中文越框；仅在解析期间局部适配 Unicode 显示宽度、
  换行与补空格，忽略 ANSI 宽度，保留分词/颜色，退出即恢复，无新增依赖。
  tyro 暂无公开字宽回调，因此使用局部内部适配。
- 六个训练配置 dataclass 统一 `#:` 单字段说明，避免普通前置注释被连续字段继承；
  34 个叶子参数均有非空、不重复说明，五个分组分别解释，保留 tyro 自动分栏。
- 根帮助与三子命令在 40–240 列共 24 组彩色检查通过；训练参数在
  40/60/80/100/120/180/240 列、彩色与无色共 14 组逐项核对，说明和边框均正常。

## T8 / T9：导出与回放（2026-11-16–17）

2026-11-16 用户授权 T6 CPU 后先推进 T8/T9，不等待 GPU，已写入 SPEC；原验收标准保留。

- ONNX 输入 `obs: fp32[1,H×93+640]`、`time_step: fp32[1,1]`；图内将 term 主序历史
  转帧主序，输出沿用部署名 `actions` 加六个动作张量。固定 batch=1、opset=18、
  单文件内嵌常量、最多 15000 帧；时间钳于 0 至全库末帧，目标窗口按所属 clip 钳位。
- 发布版元数据函数仅在 `runtime/sim.py` 调用；补全 anchor/body 与七个观测 term，
  scale/clip 等逐 term 数组同步扩展。参考导出器仅扩展名称与一组历史长度。
- **2026-11-17 去弃用**：导出由旧 TorchScript 改用 `dynamo=True`，显式依赖
  `onnxscript>=0.5.4`；常量检验按图连线而非节点名称，仍逐元素严格相等。
  两个 contextmanager 返回类型改为 `Generator[None]`，未修改数值阈值。

| CPU 导出自检 | 最大动作绝对误差（要求 <1e-4） |
|---|---|
| 初始 T8：100 次 CPU 试跑检查点、96 帧、43 组随机与边界输入 | 5.36442e-7 |
| dynamo 路径：合成动作夹具 | 7.67e-7 |
| dynamo 路径：245000 次真实动作检查点、11890 帧 | 4.17e-6 |

- GhostRollout 对 env 0 独立维护预测状态、时间、ctx 与历史；真实/预测侧各自算目标和动作，
  不用真实动作驱动虚影。两侧加 sensor 噪声、不加探索、不更新归一化；done 与周期重对齐，
  其他环境 done 不影响 env 0，周期对齐保留真实历史。默认周期沿用参考 1000000 步。
- native/viser 使用公开 `add_ghost_mesh` 薄子类与独立模型副本，RGBA 为 `[0.5,0.5,0.8,0.8]`；
  native 透明模式再乘默认 0.3，实际 alpha=0.24，绘制测试阈值保持 1e-7。
- macOS native 自动经 `mjpython` 启动，补 `sys.base_prefix/lib` 查找路径解决 libpython 问题；
  仅成功完成有限步回放后用 `os._exit(0)` 避免发生重置后的 Cocoa 退出挂起，交互模式不变。
- Mac CPU 小网络、H=3 的 100 次试跑检查点：native/viser 各 120 步（回合/周期对齐 3/21）；
  native 2000 步为 49/170，均正常退出。Viser 浏览器交互 646 步为 15/55，
  已观察参考/预测/真实机器人并验证暂停、单步、重置；图在 `outputs/mac-cpu/viser-aligned.jpg`。
  原生窗口运行过但未取得截图。试跑会摔倒，长回放有 Warp 接触容量提示
  （建议 nconmax/naconmax 36–39），未据此改默认容量，不作为物理精度或效果验收。

## 图解 Wiki（2026-11-18）

参照 `docs/wiki-posters` 的海报形式，在 `main` 按读者的问题重新编排为封面和七张横向图解。
默认环境数、训练额度和验收状态取自本版 SPEC、配置与上文实测记录。
封面使用本地参考舞步的三帧；效果页单独注明跑步片段、迭代号和评估口径。[海报与交互版入口](wiki/README.md)

八页在 320 至 1280 像素的八档视口下均无横向溢出，各页可打印为一张横向 A4。
已在浏览器核对目标窗口两端与软终止曲线的读数，禁用脚本后仍可离线阅读完整图文。
本次沿用已有 GPU 实测，T7 与最终 sim2sim 的验收状态见本文开头。

2026-11-18 修订图册用语。源码地图写明 CPU 测试如何与 SuperMimic 事先算好的结果比较，
效果页说明每次尝试统计到哪一步，包括失败的最后一步。

## 最近代码验收与复现

**最近代码验收（2026-11-18，提交前复跑）**：ruff、格式检查（51 文件）、
pyright（0 errors / warnings）、完整 pytest **72 passed，1 warning，33.22 秒**。
警告来自 mjlab 的 `torch.jit.script`；此前将弃用警告视为错误的专项检查为 67 passed，
`reportDeprecated: error` 为 0 errors。Pyright 沙箱内未解析依赖时，使用沙箱外同命令或
显式 `.venv` 的临时配置，未放宽规则。此前对拍和梯度阈值均保留。

文档整理阶段仅核对结构、链接与命令语法；提交前另完成上述代码检查。
本轮未重跑 GPU 验收，未改参考仓库或数值夹具。
通用安装、CPU 调试、续训、导出和回放命令见 [README.md](../README.md)，不在各任务重复。

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv run python tools/make_fixtures.py --reference ../supermimic
uv run python tools/check_sim.py --device cuda:0 --num-envs 2048 --steps 1000
uv run python tools/check_sim.py --device cuda:0 --num-envs 4096 --steps 1000
uv run python tools/make_motion.py --all
```

### 待用户在 RTX 5060 Ti 上运行

以下新默认规模验收应检查时间戳运行目录中的 `checkpoints/00001000/`、
`checkpoints/00001001/`、1001 行 `metrics.jsonl` 及 PNG/SVG 曲线；同时验证整卡峰值 <15 GB。
先在另一终端启动显存采样，训练退出后 Ctrl+C；重测时更换采样文件名。

```bash
mkdir -p runs
nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits \
  -lms 200 > runs/default-4096-vram-mib.csv
```

```bash
uv run nanomimic train --motion-file data/cache/lafan1-g1-v1/all.npz \
  --device cuda:0 --num-envs 4096 --output runs/g1 --max-iterations 1001
uv run python tools/evaluate_tracking.py \
  --checkpoint runs/t7-lafan/checkpoints/00245000 --num-envs 64 --device cuda:0 --seed 0
```

T7 收敛与参考比较后，须用其合格检查点做最终 sim2sim。下面首行替换为合格检查点；
ONNX 在 CPU 推理、物理在 GPU，从各片段首帧连续推进，保留摔倒终止，仅延长超时。
该脚本尚未执行，CPU 自检与试跑回放不能替代它。

```bash
export NANOMIMIC_CHECKPOINT=runs/t7-lafan/checkpoints/00245000
uv run nanomimic export --checkpoint "$NANOMIMIC_CHECKPOINT" --out /tmp/nanomimic-t8.onnx --verify
uv run python - <<'PY'
import os
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

cfg = load_config(Path(os.environ["NANOMIMIC_CHECKPOINT"]))
assert cfg.motion_file is not None
session = ort.InferenceSession("/tmp/nanomimic-t8.onnx", providers=["CPUExecutionProvider"])
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
                "obs": packed, "time_step": np.array([[frame]], dtype=np.float32),
            })[0]
            obs, _, terminated, truncated = sim.step(torch.as_tensor(action, device=sim.device))
            assert not (terminated | truncated).any(), f"第 {clip} 段第 {frame} 帧失败"
            history.push(add_noise(obs.sensor))
        print(f"第 {clip} 段完整跟踪通过，共 {end - start + 1} 帧")
        start = end + 1
PY
```

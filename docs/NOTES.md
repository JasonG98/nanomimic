# NOTES

记录实测数据、与参考仓库（HybridRobotics/supermimic）的差异、以及等待用户确认的事项。
按任务追加，最新内容放在最前面。

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

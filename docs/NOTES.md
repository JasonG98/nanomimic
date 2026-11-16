# NOTES

记录实测数据、与参考仓库（HybridRobotics/supermimic）的差异、以及等待用户确认的事项。
按任务追加，最新内容放在最前面。

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

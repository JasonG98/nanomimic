# AGENTS.md

## 项目

nanomimic：独立重写 SuperMimic 的训练管线。用可微世界模型在「想象」中反向传播，训练人形（Unitree G1）全身动作跟踪策略，并导出 ONNX 供真机部署。

**`docs/SPEC.md` 是唯一的事实来源。** 动手前先读它的第 0、1、3 节；做具体任务时读对应的模块小节。本文件只管工作方式；开发文档统一放在 `docs/`，其中 `docs/NOTES.md` 记录实测与决策。

## 工作方式

1. 按 `docs/SPEC.md` 第 5 节的顺序做任务（T0 到 T9），每个任务通过验收后再做下一个，每个任务一个提交（提交需当次授权，见第 8 条），提交信息写成 `feat(T03): model.py 与对拍夹具`。
2. 参考仓库 HybridRobotics/supermimic 只读。克隆到本仓库之外（例如 `../supermimic-ref`），只用来取常量、生成对拍夹具、核对 mjlab 契约。不要复制它的文件结构、类层次或代码。
3. 规格没写清、或与参考仓库冲突时：以 `docs/SPEC.md` 为准；细节不确定就选最简方案，并在 `docs/NOTES.md` 记一条（日期、问题、你的选择、理由）。不要自行增加功能。
4. 没有 GPU 的环境里，不要声称做过 GPU 验证。需要 GPU 的验收（T1、T6 的 GPU 部分、T7、T8 的 sim2sim）在 `docs/NOTES.md` 里标为「待用户在 RTX 5060 Ti 上运行」，并写出可直接复制的运行命令。

## 命令

T0 完成后应当可用：

```
uv sync                         # 安装依赖
uv run ruff check .             # 静态检查
uv run ruff format --check .    # 格式检查
uv run pyright                  # 类型检查，必须无报错
uv run pytest                   # 单测，CPU 即可运行
uv run nanomimic train          # 训练（需要 GPU）
uv run nanomimic play --checkpoint <dir> [--viewer native|viser]
uv run nanomimic export --checkpoint <dir> --out <file.onnx> [--verify]
```

提交前至少跑通：ruff、pyright、pytest。

## 代码规范

- 单文件不超过约 300 行；超过就拆，或检查是否在做规格之外的事。
- 公共函数有类型标注；配置用 frozen dataclass，经 tyro 暴露。
- 不用 TensorDict，批数据用 dataclass；不用全局状态；优先纯函数。
- 每个文件开头用 3 到 5 行说明职责与不变量；张量形状写在注释里（B 批、T 时间、J 关节、N 目标帧、H 历史帧）。
- 导入规则：`model.py`、`policy.py`、`sensor.py`、`imagine.py`、`buffer.py` 不得导入 mjlab、mujoco、wandb；只有 `sim.py` 与 `viewer.py` 可以导入 mjlab。这条由测试强制检查。
- 数值敏感处（积分、FK、奖励、软终止、Bryson 权重）必须有对拍夹具测试，误差小于 1e-5（fp32）。

## 不要做

- MaxCoord 与解析 IK、velocity 任务与 latent BC、多任务、ReturnSampler、手写 CUDA graph、benchmark、W&B 产物与动作库注册表、PPO、真机运行时。
- 依赖 mjlab 的私有 fork。只用发布版；缺的接口在 `sim.py` 里用子类补。
- 手改 `tests/fixtures/` 下的夹具。需要更新时用 `tools/make_fixtures.py` 重新生成，并在 `docs/NOTES.md` 说明原因。
- 为了让测试通过而放宽误差阈值或删测试。

## 完成的定义

一个任务算完成，当且仅当：

1. `docs/SPEC.md` 第 5 节该任务的验收项全部满足（GPU 项按上面第 4 条标注）；
2. ruff、pyright、pytest 通过；
3. `docs/NOTES.md` 已更新（实测数据、与参考仓库的差异、待用户确认的事项）；
4. 一个提交（经当次授权后执行，见第 8 条），信息格式见「工作方式」第 1 条。

## 需要用户确认的事项

- 第 6 节第 2 条的效果阈值：与参考实现对比后写入 `docs/NOTES.md`，等用户确认。
- 任何要偏离 `docs/SPEC.md` 的决定。

## 7. 沟通准则

凡能用中文表达的内容，一律使用中文。

- 注释与文档：代码注释、文档字符串（docstring）、项目文档、任务说明和测试说明均使用中文。
- 交互与输出：agent 回复、进度说明、程序日志、终端提示、警告、错误信息及命令行帮助等可控的自然语言输出均使用中文。
- 必要原文：代码标识符、公共 API、命令、文件路径、配置键、机器读取的字段名，以及标准要求保留的名称和格式维持原文；其解释使用中文。引用外部原文或第三方原始输出时保留原样，必要时补充中文说明。

以格赖斯合作原则及其四项会话准则（数量、质量、关系、方式）为指导：

- 数量：信息充分而不冗余。
- 质量：论述真实且有依据，不臆测、不编造。
- 关系：回应紧扣问题，不跑题。
- 方式：表达清晰有序，避免含糊。

在不牺牲准确性与必要深度的前提下，提高沟通效率。

## 8. Git 提交与推送

- 提交和推送分别按当次授权，未经当次明确授权不得自行执行。
- 不改写 Git 历史（不 rebase 已推送提交、不 amend 已推送提交、不 force push）。
- 提交信息用 Conventional Commits（feat:/fix:/docs:/test:/refactor:/chore: 等），任务卡编号放在 scope 或正文，如 feat(T03): ...。
- 中文正文按 75 字符换行。
- 提交前核对身份（user.name / user.email），并在提交信息末尾添加实际模型的 Co-authored-by 行。

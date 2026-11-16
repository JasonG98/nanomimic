# nanomimic 开发规格

2026-11-15 · 面向 coding agent · 目标设备 RTX 5060 Ti 16GB · 只含接口与规格，不含实现

**一句话**：独立重写 SuperMimic 的训练管线（可微世界模型 + 想象中反向传播，训练人形全身动作跟踪策略），代码约 1.9k 行，面向 G1 真机部署，可在 RTX 5060 Ti 16GB 上训练。

## 0. 给 coding agent 的规则

1. 参考仓库 HybridRobotics/supermimic **只读**：只用来取常量、生成对拍夹具、核对 mjlab 契约。不要复制它的文件结构、类层次或代码。本规格与参考仓库冲突时以本规格为准，并在 docs/NOTES.md 记一条。
2. 规格没写清的细节：先看各节「参考」指向的文件；仍不确定就选最简方案并记入 docs/NOTES.md，不要自行增加功能。
3. 不要实现：MaxCoord 与 IK、velocity 任务与 latent BC、多任务、ReturnSampler、手写 CUDA graph、benchmark、W&B 产物与动作库注册表、PPO、真机运行时。
4. 按第 5 节的顺序做任务，每个任务通过验收后再做下一个，每个任务一个提交。

## 1. 技术栈与规范

- Python 3.14 + uv（mjlab 或 Warp 不支持时退回 3.13）；PyTorch 2.15（CUDA 12.8 以上构建）；mjlab 1.x + MuJoCo Warp，只用发布版；onnx + onnxruntime；tyro。
- 工具：ruff、pyright（无报错）、pytest。
- 规范：单文件不超过约 300 行；公共函数有类型标注；配置全部用 frozen dataclass 并经 tyro 暴露；不用 TensorDict，批数据用 dataclass；不用全局状态；每个文件开头用 3 到 5 行说明职责与不变量；张量形状写在注释里（B 批、T 时间、J 关节、N 目标帧、H 历史帧）。
- 导入规则：model、policy、sensor、imagine、buffer 不得导入 mjlab、mujoco、wandb（由测试检查）；只有 sim.py 与 viewer.py 导入 mjlab。

## 2. 仓库结构

```
nanomimic/
├─ pyproject.toml   uv 项目配置；[project.scripts] 提供 nanomimic 命令
├─ src/nanomimic/
│  ├─ model.py     表示 + 积分 + FK + 动力学            约 270 行
│  ├─ sensor.py    proprio 合成 + 噪声 + HistoryRing     约 120 行
│  ├─ policy.py    Actor 与 Critic                       约 130 行
│  ├─ imagine.py   想象 rollout + 奖励 + 软终止          约 220 行
│  ├─ buffer.py    环形缓冲 + 窗口采样                   约 140 行
│  ├─ train.py     训练循环 + 优化器 + 检查点             约 270 行
│  ├─ sim.py       mjlab 适配                            约 310 行
│  ├─ export.py    ONNX 导出与自检                       约 150 行
│  ├─ viewer.py    世界模型虚影                          约 180 行
│  └─ cli.py       train / play / export                 约 100 行
├─ tests/          CPU 可跑的单测；tests/fixtures/ 存对拍夹具
├─ tools/          make_fixtures.py：在装有参考仓库的环境里生成夹具
├─ docs/
│  ├─ SPEC.md      本规格
│  └─ NOTES.md     实测数据与和参考仓库的差异记录
└─ AGENTS.md       coding agent 的工作方式（留在根目录，便于自动加载）
```

uv 的 src-layout：可导入包是 `src/nanomimic/`，单测与工具脚本放在仓库根目录。

## 3. 核心约定

J 为关节数（G1 是 29），动作维度 A = J，E 为并行环境数。

| 名称 | 定义 |
|---|---|
| full_state | 13 + 2J 维：根位置 3、根四元数 4、关节位置 J、根线速度 3、根角速度 3、关节速度 J（布局以参考仓库 representations.py 为准） |
| rep_state | 世界模型状态：[根高度 1, 机体系重力方向 3, 机体系线速度 3, 机体系角速度 3, 关节位置 J, 关节速度 J]（以参考仓库为准） |
| aux_state | 根位姿 7 维（位置 3 + 四元数 4），由积分得到，不由网络预测 |
| 预测 | [Δ线速度 3, Δ角速度 3, Δ关节速度 J]，共 6 + J，是机体系下加速度级的增量 |
| wm_goal | [content, rel_pos_b, rel_ori_b]；content 是 N 帧的 [关节位置, 关节速度]；rel_pos_b 为 3N 维；rel_ori_b 为 6N 维 |
| actor_goal | [content, rel_ori_b]，即去掉 rel_pos_b |
| 目标窗口 | N = 10 帧，间隔 5 个控制步，越过当前 clip 末帧的取末帧 |
| proprio 单帧 | [投影重力 3, 基座角速度 3, 关节位置 J, 关节速度 J, 上一动作 A]，加均匀噪声，半宽依次为 0.05、0.2、0.01、0.5、0 |
| Actor 输入 | H 帧 proprio（帧主序）拼接 actor_goal；H 即 history_length |

## 4. 模块规格

### 4.1 sim.py

- 基于 mjlab 的 G1 平地动作跟踪任务做修改并注册：观测组提供 full_state、goal、actor（proprio，关闭 mjlab 自带噪声，噪声统一由 sensor.py 加）；随机化事件只保留脚摩擦、质心偏移、编码器偏置三项；去掉 self_collisions 奖励与 ee_body_pos 终止；joint_limit 权重设为 -2.0。参考：tasks/env_cfgs.py。
- 重置采样用 mjlab 自带的 adaptive 模式（实现前核对名称与可用性）。
- 对外只暴露一个 Sim 类：reset()、step(action)（返回 Obs、reward、terminated、truncated），以及 num_envs、device、robot、motion、ctx_spec、read_ctx()、motion_time。
- Obs 含 full_state、goal、sensor（干净的 proprio）。
- robot（RobotSpec）：关节名、默认关节位置、软限位、动作缩放与偏置、控制步长 dt、body 名、锚点 body、FK 所需的父子关系、关节轴与固定旋转，以及软终止的两个阈值（取自任务配置里 anchor_pos 与 anchor_ori 终止项）。
- motion（Motion）：预计算的参考张量 body_pos_w、body_quat_w、body_lin_vel_w、body_ang_vel_w（[T, B, ·]）、joint_pos、joint_vel（[T, J]）、各 clip 末帧索引、总帧数。动作文件用本地 npz 路径加载。
- ctx_spec：三个随机化通道及取值范围；read_ctx() 返回归一化到 [-1, 1] 的 [E, C]。

### 4.2 model.py

- 纯函数加一个 Dynamics 模块，不导入仿真器。函数：represent、aux_state、integrate、joint_state、reconstruct_full_state、world_bodies（FK）、loss_groups。
- 积分：把预测的增量加到速度上，再用更新后的速度半隐式积分位姿；步长为控制步长 dt（substep 系数 1.0）。精确形式参考 representations.py 的 GenCoord 部分与 utils/gencoord_helpers.py。
- Dynamics：MLP 3 层 × 2048（Linear、LayerNorm、GELU）；输入 [归一化 rep_state, 归一化的处理后动作, ctx]；输出预测。处理后动作 = 原始动作 × 缩放 + 偏置。state 与 action 各一个 RunningNorm，用新采集的数据在线更新。
- 损失分组（Bryson 规则，权重 = 1 / 名义误差²）：root_pos 0.01，root_ori 0.02，joint_pos 0.1，joint_vel 3.14；root_ori 用旋转矩阵乘 sqrt(0.5) 作为平方误差通道。

### 4.3 sensor.py

- synth_proprio(rep_state, aux_state, last_action)：由状态合成 proprio 单帧（投影重力由根四元数得到）；add_noise 按第 3 节的半宽加均匀噪声，是采集与想象共用的唯一噪声入口。
- HistoryRing(H)：push(frame)、reset(rows, frame)、window() 返回 [B, H, D] 帧主序。被重置的行用同一帧（同一次噪声）填满历史。

### 4.4 policy.py

- Actor：编码器（actor_goal → 32 维，tanh）加解码器（[潜变量, 展平的 proprio 历史] → 动作）；解码器隐藏层 2048、1024、512、256；观测与目标各有 RunningNorm。其余细节参考 models.py。
- Critic：MLP 256、128；输入 [rep_state, actor_goal, ctx]（特权信息）。

### 4.5 imagine.py

- imagine(dynamics, actor, critic, motion, start, horizon, cfg) 返回逐步 reward、终止概率、critic 观测，以及末端观测。纯函数，状态显式传递，全程可微。
- 单步顺序：取 wm_goal → Actor 出动作并加探索噪声 → dynamics 推进 → FK 得世界系 body → 读参考 → 软终止与奖励 → 动作库时间加 1（在 clip 末帧钳住）→ 更新 HistoryRing。
- 奖励 = dt × (Σ 各项 w × exp(−err / std²) − 0.1 × ‖a − a_prev‖² − 2.0 × 关节越限量)。参考：utils/tracking_reward.py。

| 项 | err | w | std |
|---|---|---|---|
| 锚点位置 | 根 body 世界位置的平方误差 | 0.5 | 0.3 |
| 锚点朝向 | 根 body 旋转误差角的平方 | 0.5 | 0.4 |
| 相对位置 | 重锚定后各 body 位置平方误差的均值 | 1.0 | 0.3 |
| 相对朝向 | 重锚定后各 body 旋转误差角平方的均值 | 1.0 | 0.4 |
| 线速度 | 各 body 线速度平方误差的均值 | 1.0 | 1.0 |
| 角速度 | 各 body 角速度平方误差的均值 | 1.0 | 3.14 |

- 重锚定：目标 body 位姿表示在这样一个坐标系里：xy 与偏航取自预测的根，z 与滚转俯仰取自目标的根。关节越限量 = Σ(下限 − q)₊ + (q − 上限)₊，用软限位。
- 软终止：归一化误差 = max(锚点高度误差 / 阈值_pos, 倾斜误差 / 阈值_ori)；终止概率 p = sigmoid((归一化误差 − 1) / 0.05)。参考：envs/world_model_env.py 的软终止函数。
- 回报：存活概率 s_{t+1} = s_t × (1 − p_t)；折扣回报 = Σ γᵗ × s_t × r_t；末端加 γᴴ × s_H × V(末端)，V 钳在 max_step_reward / (1 − γ) 以内，其中 max_step_reward = 5 × dt（正权重之和 5.0 乘 dt）。
- 显存：每个想象步用 torch.utils.checkpoint 包起来，配置里可关闭。

### 4.6 buffer.py

- 每个环境一个环形缓冲，容量 1000 步；字段：full_state、sensor、action、motion_time、terminated、truncated、ctx。
- sample_window(L, B)：返回长度为 L、内部不跨越 done 的窗口，按环境分层采样，用累加和向量化。
- sample_start(B, H)：为策略想象取起点：随机一帧的 full_state、motion_time、ctx，以及它之前 H − 1 帧的 sensor 与 action。越过 episode 起点的槽位用 episode 首帧填充，并带掩码；用来热启动 HistoryRing。

### 4.7 train.py

- Trainer 只做排程，三个阶段是三个小对象：Collector（用 Actor 加噪声在真实仿真里走一步，写缓冲，向归一化器提交新数据，对 sensor 加噪声）、WorldModelFitter、PolicyImprover。
- 每次迭代：Collector 走一步；训练次数 = replay_ratio × num_envs / (wm.num_batches × wm.batch × wm.horizon)，小数部分累积到下次；每次训练先 fit 再 improve。
- WorldModelFitter：取 num_batches 个窗口，从首帧开始用真实动作自回归 rollout horizon 步，用 loss_groups 加权平方误差，梯度范数裁剪到 1.0。
- PolicyImprover：从 sample_start 取起点调用 imagine；Actor 的损失 = −（折扣回报 + 末端 bootstrap）的均值；Critic 用 GAE（lambda 0.95）回报训练，5 轮 × 4 个 minibatch；梯度做范数裁剪并加逐元素裁剪 0.1；Adam beta2 取 0.95。
- 优化器：每个 MLP 的中间隐藏层权重用 torch.optim.Muon，其余参数（首末层、偏置、LayerNorm）用 Adam；学习率 2e-4。
- 编译：对 rollout 与损失函数用 torch.compile；不要手写 CUDA graph。
- 检查点：每 1000 次迭代保存 world_model.pt、actor.pt、critic.pt（含归一化状态）与 config.json（带 schema_version），支持 --resume；日志每 10 次迭代打印，W&B 是可选的 Sink（只记标量）。

### 4.8 export.py

- 导出 motion 契约的 ONNX，沿用 mjlab 跟踪任务的部署契约：输入 (obs, time_step)，输出 (action, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w)。
- 动作库张量作为常量 buffer 嵌进图里，time_step 钳在末帧；obs 先在图内从 term 主序转成帧主序再喂 Actor。obs 布局与元数据以参考仓库 runners/exporter.py 的 motion 部分为准，差异记入 docs/NOTES.md。
- 元数据：mjlab 的 get_base_metadata 与 attach_metadata_to_onnx，另加 anchor_body_name、body_names、observation_names、observation_history_lengths。
- 仅当动作库不超过 15000 帧时导出。export --verify 执行自检。

### 4.9 viewer.py

- GhostRollout 只对 env 0 维护一条世界模型闭环预测。每步：真实侧，Actor 吃真实 sensor 与 goal，再 env.step；世界模型侧，用预测状态合成 sensor（带噪声）与 wm_goal，Actor 出动作，dynamics 推进；再用 reconstruct_full_state 取根位姿与关节位置，拼成 qpos 画虚影。
- 重新对齐：回合结束或每 num_steps_rollout 步，用真实状态重置预测状态与历史（默认值取参考仓库）。
- 渲染：复制一份 MuJoCo 模型，geom 颜色设为 [0.5, 0.5, 0.8, 0.8]，用 mjlab viewer 的 ghost 接口叠加。原生窗口与 viser 窗口各一个薄子类，只改单步与可视化回调。参考：scripts/viewers.py。

### 4.10 cli.py

train；play --checkpoint [--viewer native|viser]；export --checkpoint --out [--verify]。

## 5. 开发任务

| 任务 | 内容 | 验收 |
|---|---|---|
| T0 | 脚手架：uv 项目、ruff 与 pyright 与 pytest 配置、导入规则测试、docs/NOTES.md | 全部检查通过 |
| T1 | 在 5060 Ti 上跑通 mjlab G1 跟踪任务，实测 2048 与 4096 环境的显存和每秒步数，写入 docs/NOTES.md | 确定 num_envs 默认值 |
| T2 | sim.py 与 tools/make_fixtures.py（在参考仓库环境里生成夹具） | 随机动作跑 100 步；RobotSpec、Motion、ctx 的形状断言通过 |
| T3 | model.py | 与夹具对拍：represent、integrate、FK、loss_groups 误差小于 1e-5 |
| T4 | sensor.py、buffer.py | 窗口内无 done；掩码正确；HistoryRing 重置语义正确；噪声在半宽内 |
| T5 | policy.py、imagine.py | 奖励与软终止对拍误差小于 1e-5；Actor 参数梯度有限且非零；开关逐步 checkpoint 前后梯度一致 |
| T6 | train.py、cli train | CPU 合成数据上世界模型损失下降；GPU 上 50 次迭代无 NaN，显存峰值低于 15GB |
| T7 | 完整训练与效果验证 | 第 6 节第 2 条 |
| T8 | export.py | 第 6 节第 3 条 |
| T9 | viewer.py、play、README | 虚影随仿真运行并定期重新对齐 |

## 6. 验收

1. 单元：T3 与 T5 的对拍夹具全部通过。
2. 效果：同一段动作（lafan_run1_subject5）训练到收敛，在仿真中完整跟踪、不摔倒的成功率和平均跟踪误差达到参考实现的水平；不要求训练曲线一致。阈值与参考实现对比后写入 docs/NOTES.md，由用户确认。
3. 导出：onnxruntime 与 PyTorch Actor 的动作差小于 1e-4；用 ONNX 在仿真里闭环跑完整段动作且不摔倒（sim2sim）。
4. 资源：默认配置下显存峰值低于 15GB。
5. 规范：ruff 与 pyright 无报错；单文件不超过约 300 行；导入规则测试通过。

## 7. 默认配置

| 项 | 值 |
|---|---|
| num_envs | 2048（T1 实测后可调） |
| 缓冲 | 每环境 1000 步 |
| replay_ratio | 16 |
| 世界模型 | lr 2e-4，batch 2048，窗口 8，每次 10 个 batch，隐藏层 3 × 2048，梯度范数裁剪 1.0 |
| 策略 | lr 2e-4，batch 2048，想象 32 步，探索噪声 0.1，gamma 0.99，lambda 0.95，Adam beta2 0.95，逐元素梯度裁剪 0.1 |
| Critic | 256、128；5 轮 × 4 个 minibatch |
| 精度 | fp32 参数 + TF32 矩阵乘 |
| history_length | 取参考仓库的默认值 |
| 日志与检查点 | 每 10 次迭代打印，每 1000 次迭代保存 |

## 8. 风险与待确认

1. 16GB 的显存数字是估算，以 T1 与 T6 的实测为准，必要时降低 batch 或环境数。
2. mjlab 的 adaptive 采样名称与发布版 API 在 T2 核对；缺失时在 sim.py 内用子类补，不要依赖私有 fork。
3. 原实现的域随机化只有三项，没有动作延迟与外力扰动；首次上真机前必须先通过 sim2sim 自检，再用吊装与低增益逐步放开。
4. 真机运行时（读取 ONNX、对接机器人 SDK、安全限幅）不在本项目内；ONNX 沿用 mjlab 跟踪任务的部署契约。
5. 这是独立重写，不是严格意义的 clean-room；沿用 Apache-2.0，并保留对 SuperMimic 的致谢。

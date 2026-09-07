# 中期检查报告内容 & 大脑层完整实施计划

---

## 第一部分：毕业设计工作中存在的问题及解决思路

### 问题一：SolidWorks → URDF 的模型适配问题

**问题描述：**
机器人结构由 SolidWorks 建模后导出 URDF 用于 PyBullet 仿真。导出过程中出现多类严重适配问题：
1. SolidWorks 的坐标系约定与 URDF/PyBullet 的坐标系之间的旋转变换不一致，导致每个 steering joint 的 URDF origin RPY 出现复杂的三维旋转（如 `-1.57, 1.57, 1.57`），使得四个转向电机的物理朝向（零位基准）各不相同。
2. 碰撞体 (collision mesh) 的包围盒在简化导出后与视觉模型 (visual mesh) 不匹配，导致仿真中出现穿模或多余碰撞。
3. 惯性参数 (inertia tensor) 在自动计算时与实际零件差异较大，影响物理仿真的真实感。

**解决思路：**
1. **IMU 式标定方案**：放弃在代码中硬编码每个 steering joint 的 RPY 偏移量（因为不同构型下偏移量会变化），转而采用类似 IMU 的实时测量方案——在 `reset()` 时执行一次性标定，直接读取每个 steering link 在世界坐标系下的实际朝向（通过旋转矩阵投影提取 yaw），减去当前关节角度即得到 `zero_yaw` 基准，再加上几何修正常数。此方案完全消除了对 URDF RPY 的人工依赖，且在形变过程中自动适应（见 `robot_env.py` 中 `_calibrate_imu_offsets()` 函数）。
2. **碰撞白名单/黑名单机制**：针对可变拓扑结构中不同模块之间的碰撞关系，在 `_setup_collisions()` 中建立了精细化的碰撞过滤规则，对相邻模块对允许碰撞，对不可能物理接触的模块对关闭碰撞检测。
3. **手动校正惯性参数**：针对关键运动部件（轮子、转向连杆）手动设置了合理的接触刚度（10000）、阻尼（500）和摩擦系数，以保证仿真行为接近真实物理。

---

### 问题二：可变拓扑导致的"空间整合"与坐标系问题

**问题描述：**
传统固定结构的机器人只有一套固定的轮位坐标。然而本课题的机器人具有可变拓扑特性——当机器人从 O 形变形为 8 形时，四个轮子相对于机体中心的位置、朝向和接地点全部发生变化。这带来了两个核心困难：
1. **机体坐标系失效**：传统方案在机体坐标系 (Body Frame) 下做运动学解算，但可变拓扑下"机体坐标系"的定义本身就在漂移——base link 的质心在形变过程中会偏离四轮几何中心，导致速度反馈和运动学解算出现系统性偏差。
2. **舵轮零位基准漂移**：形变改变了每个转向电机的安装基座朝向，原有的静态偏置角 (steering offset) 在变形后不再正确，导致轮子指向混乱。

**解决思路：**
1. **统一世界坐标系**：彻底放弃"机体坐标系"概念。所有运动学解算（`swerve_kinematics.py`）、速度反馈（`get_world_velocity()`）、MPC 规划（`mpc_controller.py`）全部在世界坐标系下进行。轮位相对于"四个转向柱的世界坐标几何中心"计算，而非相对于 base link。
2. **动态标定转向基准**：每帧实时测量 steering link 的世界朝向（IMU 式），而非依赖预设的偏置表。`get_steering_zero_yaws_imu()` 函数实现了这一动态标定，配合 EMA 低通滤波（`ZeroYawSmoother`）消除 PyBullet 高频物理噪声，确保转向解算的持续准确。
3. **虚拟速度探针**：在 `get_world_velocity()` 中，使用刚体运动学投影公式 `V_center = V_base + w x (Center - Base)` 将 base link 的速度读数修正到四轮几何中心，解决质心偏移导致的速度误差。

---

### 问题三：GNN 模型输入空间设计与训练

**问题描述：**
早期版本的 GNN 采用"离散 Mode + 连续 Rate"的双重输入方案（14 维输入），存在以下问题：
1. 离散模态之间的切换边界不连续，GNN 无法学习到平滑的跨模态过渡。
2. 训练数据需要为每种模态单独采集，数据管理复杂。
3. 前期训练（`run_20260110` 等早期实验）的 Val Loss 高达 0.459，模型仅过拟合了训练轨迹，泛化性差。

**解决思路：**
1. **连续拓扑流形 Phi 重构**：将 4 种离散构型（8形、O形、门形、1形）统一映射到一条连续参数 Phi 从 0 到 3 上，输入维度从 14 维降至 10 维 `[7D one-hot ID | Phi_norm | is_locked | base_angle]`。GNN 的输入空间从"离散乘以连续"变为"纯连续"，天然支持跨构型的平滑插值。
2. **硬门控机制 (Hard-Gating)**：在 GNN 输出端引入硬门控——锁死关节直接输出预设的 base_angle，可动关节输出 GNN 的网络预测值。公式为 `Final = Raw * (1-is_locked) + base_angle * is_locked`。此设计保证了拓扑约束的物理可行性。
3. **边界加权训练策略**：在损失函数中引入边界加权（Phi=0, 1, 2, 3 附近样本权重指数增大）和辅助损失（让 raw 网络输出也对齐目标角度），最终模型在连续 Phi 版本上达到 Val Loss = 0.00216、MAE = 0.026 rad（约 1.5 度），相较早期提升两个数量级。
4. **Phi=2 边界不连续问题**：发现 O_shape(rate=0) 到 door_shape(rate=0) 在 Phi=2 处存在物理状态跳变（关节角度不连续约 pi/2）。硬门控机制分别保证了两侧输出的正确性，但过渡区域需要 GNN 学习快速切换，是目前已知的局限性。

---

### 问题四：变形过程中的运动稳定性控制

**问题描述：**
机器人在边移动边变形时，以下因素导致运动极不稳定：
1. 变形过程中机体惯量、质心位置和轮-地接触力分布实时变化，固定 PID 增益无法适应。
2. PyBullet 物理引擎的被动关节（pivot joints）在形变时弹回零位，导致转向柱离地。
3. 舵轮转向在正负 180 度边界跳变，触发 position control 瞬间最大力矩导致机器人飞天。

**解决思路：**
1. **Phi-aware 增益调度**：PID 控制器的增益根据当前 Phi 值动态调整——紧凑形态（Phi 远离 1.5）惯量小、增益放大；展开形态（Phi 约 1.5）惯量大、增益缩小（见 `pid_controller.py` 中 `update_gains_for_phi()`）。
2. **Pivot 棘轮机制**：在 `update_pivot_hold()` 中，每物理 tick 将 pivot 关节的 position control 目标更新为其当前实际角度，使 pivot 变为"只能被外力推动但不会弹回"的棘轮，解决了结构变化时 pivot 弹回导致舵轮离地的问题。
3. **增量式转角输出**：在 `swerve_kinematics.py` 中，`_optimize_action()` 方法采用增量式角度更新 `final_angle = current + error`，而非直接输出归一化后的目标角度。这保证了 steering joint 的角度输出在 +-pi 边界处连续无跳变。此外在每帧同步 `swerve.last_angles` 到物理引擎实际读数，防止控制量脱节。

---

### 问题五：MPC 轨迹跟踪的机体/世界坐标系转换

**问题描述：**
KinematicMPC 基于全向 Unicycle 运动学模型工作在机体坐标系下（输出 `[vx_body, vy_body, omega]`），但底盘逆解库（`SwerveDriveController`）需要世界坐标系下的速度输入。早期版本未做坐标转换，导致在机器人偏航角较大时，MPC 给出的速度方向与实际执行方向偏差 90 度以上。

**解决思路：**
在 MPC 输出端增加从机体系到世界系的旋转变换：
```
Vx_world = vx_body * cos(yaw) - vy_body * sin(yaw)
Vy_world = vx_body * sin(yaw) + vy_body * cos(yaw)
```
同时在 `test_mpc_pybullet.py` 和 `run_mpc_gui.py` 中均实现了此转换，并通过实际 S 曲线跟踪实验验证了修正效果。

---

## 第二部分：下一阶段工作计划与研究内容

### 总体目标

在已完成的底层脊髓（GNN）和中层小脑（MPC+PID）基础上，实现上层大脑的三种方案（规则大脑、DRL 大脑、NavFormer 大脑），在统一的实验场景下进行定量对比，形成完整的三层控制架构论文。

### 时间线（约 5 周）

```
Week 1 (4/9 - 4/15)    基础设施 + 规则大脑
Week 2 (4/16 - 4/22)   DRL 大脑 (PPO/SAC)
Week 3 (4/23 - 4/29)   NavFormer 大脑
Week 4 (4/30 - 5/6)    对比实验 + 消融实验 + 可视化
Week 5 (5/7 - 5/13)    论文撰写 + 答辩准备
```

---

### Week 1：实验基础设施 + 规则大脑 (Scripted Brain)

#### 1.1 统一实验环境搭建（2天）

**目标：** 构建一个可复用的 benchmark 场景，所有三种大脑在同一场景下运行并评估。

**场景设计：**

| 场景 | 描述 | 考核重点 |
|------|------|----------|
| **Scene A: 窄门穿越** | O形态导航至窄门前 --> 变形为1形态 --> 穿过宽 0.15m 的窄缝 --> 恢复O形态 | 形变决策时机 + 变形精度 |
| **Scene B: 多目标巡航** | 依次到达 3-5 个路标点，无需变形 | 导航精度 + 路径规划 |
| **Scene C: 障碍规避+变形** | 在有障碍物的区域中从 A 点到 B 点，需要多次变形 | 综合决策能力 |

**构建内容：**
- `src/envs/benchmark_env.py`：封装 PyBullet 场景（墙壁、窄门、障碍物）
- `src/envs/task_manager.py`：统一的任务定义接口（起点、终点、检查点、奖励）
- `scripts/evaluate_brain.py`：标准化评估脚本，输出统一指标表

**评估指标体系：**

| 指标 | 含义 |
|------|------|
| **Task Success Rate** | 是否成功完成全部任务目标 |
| **Total Time** | 从起点到终点的总消耗时间 |
| **Path Length** | 实际行走路径总长 |
| **Avg. Tracking Error** | MPC 跟踪误差的时间平均值 |
| **Morphing Count** | 触发构型切换的次数 |
| **Smoothness (Jerk)** | 加速度变化率，衡量运动平滑度 |

#### 1.2 规则大脑实现（2天）

**文件：** `src/brains/scripted_brain.py`

**核心设计：** 基于有限状态机 (FSM) 的分段决策器

```
状态: NAVIGATE -> APPROACH_GATE -> MORPHING -> TRAVERSE -> RESTORE -> NAVIGATE
```

**实现要点：**
- 预编程路径点序列和对应的 Phi 目标值
- 根据距离阈值触发状态转换（如距窄门小于 0.5m 时开始变形）
- 输出格式与后续 DRL/NavFormer 一致：`(trajectory_ref, phi_target)`

#### 1.3 底层消融实验（1天）

同步完成 GNN 消融实验（无门控 / Lite / MLP / Formula 对比），为论文第四章积累数据。

---

### Week 2：DRL 大脑 (PPO/SAC)

#### 2.1 Gym 环境封装（1天）

**文件：** `src/envs/brain_gym_env.py`

将现有的 `MorphingRobotEnv` 包装为标准 Gymnasium 接口，供 Stable-Baselines3 训练：

| 设计要素 | 定义 |
|----------|------|
| **观测空间** | `[x, y, yaw, vx, vy, omega, Phi_current, dist_to_goal, angle_to_goal, gate_width, gate_dist]` (约 15D) |
| **动作空间** | `[vx_cmd, vy_cmd, omega_cmd, Phi_target]` (4D 连续) |
| **奖励函数** | `r = r_progress + r_collision + r_time + r_morphing_penalty + r_success` |

**奖励设计细节：**
- `r_progress`：每步靠近目标的距离增量（正奖励）
- `r_collision`：碰撞惩罚（负 10）
- `r_time`：每步小额时间代价（负 0.01）
- `r_morphing_penalty`：每次变形的小代价（负 0.5），鼓励必要时才变形
- `r_success`：到达终点的大奖励（正 100）

#### 2.2 PPO/SAC 训练（3天）

**文件：** `scripts/train_drl_brain.py`

```python
from stable_baselines3 import PPO, SAC

# 两个算法都训练，最终取效果更好的
model_ppo = PPO("MlpPolicy", env, verbose=1, ...)
model_sac = SAC("MlpPolicy", env, verbose=1, ...)
```

**训练策略：**
1. **课程学习 (Curriculum Learning)**：
   - Phase 1（50K steps）：仅直线导航，无障碍物，不需要变形
   - Phase 2（100K steps）：加入障碍物和窄门，需要变形
   - Phase 3（200K steps）：完整困难场景
2. **Domain Randomization**：随机化起点/终点位置、窄门宽度、障碍物布局
3. **预计训练资源**：350K steps，约 6-12 小时（单 GPU）

#### 2.3 DRL 评估（1天）

在三个 benchmark 场景上运行训练好的 PPO/SAC 策略，记录所有指标。

---

### Week 3：NavFormer 大脑

#### 3.1 架构设计（1天）

**文件：** `src/brains/navformer.py`

采用 **Decision Transformer** 范式——将导航+变形决策建模为序列到序列的自回归生成问题：

```
输入: [RTG_t, s_t, a_{t-1}, RTG_{t-1}, s_{t-1}, a_{t-2}, ...]  (Context Window = K steps)
输出: a_t = [vx, vy, omega, Phi_target]
```

**模型参数估算：**

| 组件 | 规模 |
|------|------|
| 状态 Encoder | MLP 15D -> 64D |
| 动作 Encoder | MLP 4D -> 64D |
| RTG Encoder | MLP 1D -> 64D |
| Transformer | 3 层, 4 heads, d=128, context K=20 |
| 输出 Head | MLP 128D -> 4D |
| **总参数量** | 约 200K - 500K |

#### 3.2 离线训练数据收集（1天）

使用**规则大脑 (Scripted Brain)** 作为专家策略，在仿真中收集大量"专家演示"轨迹：

- 每个 benchmark 场景采集 500-1000 条成功轨迹
- 数据格式：`(state, action, reward, next_state, done)` 序列
- 计算每条轨迹的 Returns-to-Go (RTG)
- 存储为统一的 HDF5 或 pickle 数据集

#### 3.3 NavFormer 训练（2天）

**文件：** `scripts/train_navformer.py`

- 离线监督学习：以收集的专家轨迹做行为克隆 (Behavior Cloning) 式训练
- 损失函数：MSE(action_predicted, action_expert)
- 训练资源：约 100 epochs，1-2 小时
- 可选：在线微调 (Online Fine-Tuning) 用 RL reward 信号迭代数据集

#### 3.4 NavFormer 评估（1天）

在三个 benchmark 场景下评估 NavFormer，与规则大脑和 DRL 进行横向对比。

---

### Week 4：全面对比实验 + 可视化

#### 4.1 三种大脑横向对比（2天）

**核心对比表（论文第六章核心图表）：**

| 指标 | Scripted Brain | PPO | SAC | NavFormer |
|------|---------------|-----|-----|-----------|
| Scene A 成功率 | - | - | - | - |
| Scene B 成功率 | - | - | - | - |
| Scene C 成功率 | - | - | - | - |
| 平均完成时间 | - | - | - | - |
| 平均跟踪误差 | - | - | - | - |
| 变形次数 | - | - | - | - |
| 运动平滑度 | - | - | - | - |
| 推理延迟(ms) | - | - | - | - |

#### 4.2 消融实验补充（1天）

| 实验 | 目的 |
|------|------|
| GNN 消融 (无门控 / Lite / MLP / Formula) | 证明 GNN + 硬门控的必要性 |
| MPC 对比 (MPC vs PID-only vs 开环) | 证明 MPC 跟踪优势 |
| Phi-aware PID vs 固定增益 PID | 证明增益调度的稳定性贡献 |

#### 4.3 可视化与演示视频（2天）

| 产出 | 说明 |
|------|------|
| GNN Phi-Sweep 曲线图 | 7 关节角度随 Phi 变化的连续曲线 |
| GNN 注意力热力图 | GAT 注意力权重在不同 Phi 下的分布 |
| MPC 轨迹跟踪对比图 | 参考 vs 实际轨迹，不同构型下 |
| 4 种大脑决策轨迹可视化 | 俯视图，含变形时刻标注 |
| PyBullet 演示视频 (3 段) | 每个 benchmark 场景的最佳方案录屏 |

---

### Week 5：论文撰写 + 答辩准备

**论文章节结构：**

| 章节 | 内容 | 页数估计 |
|------|------|----------|
| 第1章 绪论 | 背景、现状、创新点 | 5-6页 |
| 第2章 理论基础 | GATv2、MPC、DRL、Decision Transformer | 8-10页 |
| 第3章 系统架构 | 三层仿生架构、连续拓扑流形 Phi 定义、数据流 | 6-8页 |
| 第4章 脊髓层 | GNN 建模、10D 特征、硬门控、训练策略 | 8-10页 |
| 第5章 小脑层 | MPC 设计、PID 闭环、舵轮运动学 | 6-8页 |
| 第6章 大脑层 | 三种方案设计、训练过程 | 6-8页 |
| 第7章 实验 | 消融、对比、可视化 | 10-12页 |
| 第8章 总结与展望 | 总结+未来工作 | 2-3页 |
| **合计** | | **51-65页** |

---

## 风险评估与降级策略

> [!WARNING]
> 以下为可能的时间风险和对应的降级方案。

| 风险 | 概率 | 降级方案 |
|------|------|----------|
| DRL 训练不收敛或效果差 | 中 | 仅呈现 learning curve 和失败分析，对比中说明 DRL 对于变拓扑问题的局限性（这本身就是一个有价值的结论） |
| NavFormer 数据不足或效果差 | 中 | 缩小为 Behavior Cloning baseline，不做 online fine-tuning |
| 环境 bug 导致实验延迟 | 低 | 先在纯运动学测试中验证大脑逻辑，后接入 PyBullet |
| 时间不足 | 低 | 砍掉 Scene C（最复杂场景），只保留 A+B；NavFormer 可降级为"架构设计+初步结果" |

> [!IMPORTANT]
> **最坏情况下**，规则大脑 + 已完成的 GNN/MPC 已经足够支撑一篇结构完整的本科毕设。DRL 和 NavFormer 是**锦上添花的对比实验**，即使效果不如规则大脑，"分析与解释为什么效果不好"同样具有学术价值。

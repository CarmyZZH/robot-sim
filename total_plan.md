# 🧠 可变拓扑机器人三层架构 — 实现状态全景分析

## 一、架构蓝图 vs 代码实现：对照总览

| 层级 | 蓝图规划 | 实现状态 | 完成度 |
|------|----------|----------|--------|
| **底层 脊髓 (Spinal Cord)** | GNN + 连续拓扑流形 Φ + 硬门控 | ✅ **已完成** | 🟢 95% |
| **中层 小脑 (Cerebellum)** | 轻量级运动学 MPC + PID 闭环 | ✅ **已完成** | 🟢 85% |
| **上层 大脑 (Cerebrum)** | DRL / NavFormer 高级导航 | ❌ **未实现** | 🔴 0% |

---

## 二、已完成部分：逐层详解

### 🟢 底层 — 脊髓 (Spinal Cord): GNN 拓扑执行层

> [!TIP]
> 这是你的**核心创新点**，完成度最高、质量最好，也是论文最大卖点。

**已实现的核心组件：**

| 文件 | 功能 | 关键设计 |
|------|------|----------|
| [model.py](file:///e:/Carmy/TARS/robot-sim/src/models/model.py) | MorphGNN (3层 GATv2) + MorphGNNLite | 10D 输入 → 64D 隐藏 → 1D 角度输出 |
| [dataset.py](file:///e:/Carmy/TARS/robot-sim/src/models/dataset.py) | 连续相位 Φ 数据集 | CSV → PyG Data 自动转换 |
| [graph_converter.py](file:///e:/Carmy/TARS/robot-sim/src/utils/graph_converter.py) | Φ ↔ (mode, rate) 双向映射 | 连续流形 Φ∈[0,3] 统一离散构型 |
| [collect_gnn_data.py](file:///e:/Carmy/TARS/robot-sim/scripts/collect_gnn_data.py) | 训练数据收集（几何公式生成） | 3003 + 边界增强采样 |
| [train.py](file:///e:/Carmy/TARS/robot-sim/scripts/train.py) | 完整训练流水线 | 边界加权损失 + 辅助损失 + 仪表盘可视化 |

**关键创新实现：**

1. **连续拓扑流形 Φ ∈ [0, 3]：** 将 4 种离散构型（8形、O形、门形、1形）统一到一条连续曲线上
   - `Φ ∈ [0,1)` → 8_shape
   - `Φ ∈ [1,2]` → O_shape  
   - `Φ ∈ (2,3)` → door_shape
   - `Φ = 3` → 1_shape

2. **节点 10D 特征向量：** `[7D one-hot ID | Φ_norm | is_locked | base_angle]`

3. **硬门控机制 (Hard-Gating)：** 
   ```
   Final_Output = Raw_GNN × (1 - is_locked) + Base_Angle × is_locked
   ```
   确保锁死关节角度物理可行，解决"可变拓扑"问题。

4. **训练结果：** Best Val Loss = 0.00216, MAE = 0.026 rad ≈ 1.5°

---

### 🟢 中层 — 小脑 (Cerebellum): 运动学 MPC + PID 闭环

**已实现的核心组件：**

| 文件 | 功能 | 关键设计 |
|------|------|----------|
| [mpc_controller.py](file:///e:/Carmy/TARS/robot-sim/src/controllers/mpc_controller.py) | 轻量级 Kinematic MPC | N=10 步预测时域, SLSQP 求解, 热启动 |
| [pid_controller.py](file:///e:/Carmy/TARS/robot-sim/src/controllers/pid_controller.py) | 三通道速度 PID + Φ-aware 增益调度 | vx/vy/ω 独立闭环, EMA 滤波 |
| [swerve_kinematics.py](file:///e:/Carmy/TARS/robot-sim/src/controllers/swerve_kinematics.py) | 全向舵轮运动学逆解 | 世界坐标系, 倒车优化 |
| [test_mpc_pybullet.py](file:///e:/Carmy/TARS/robot-sim/scripts/test_mpc_pybullet.py) | MPC 轨迹跟踪 PyBullet 测试 | S 曲线跟踪演示 |
| [test_mpc_pure_kinematics.py](file:///e:/Carmy/TARS/robot-sim/scripts/test_mpc_pure_kinematics.py) | 纯运动学 MPC 测试 | 无物理引擎验证 |

**MPC 核心参数：**
- 预测步长 `dt = 0.05s`，预测时域 `N = 10`
- 代价函数：`Q (轨迹跟踪) + R (控制幅值) + Rd (平滑度)`
- 线速度上限 `0.4 m/s`，角速度上限 `1.5 rad/s`

**PID 闭环特色：**
- Φ-aware 增益调度：紧凑形态→高增益，展开形态→低增益
- 零速死区防震荡
- EMA 低通滤波消除 PyBullet 高频噪声

---

### 🟢 仿真环境 + GUI 集成

| 文件 | 功能 |
|------|------|
| [robot_env.py](file:///e:/Carmy/TARS/robot-sim/src/envs/robot_env.py) | PyBullet Gym 环境 (590行), URDF 加载, IMU 标定, 碰撞配置 |
| [run_mpc_gui.py](file:///e:/Carmy/TARS/robot-sim/scripts/run_mpc_gui.py) | 完整 GUI 控制器 (512行), GNN/Formula 切换, PID 开关, HUD |

---

## 三、未实现部分：上层大脑 (Cerebrum)

> [!CAUTION]
> 大脑层目前**完全空缺**。当前上层是 GUI 滑块手动控制，没有任何自主导航能力。

蓝图规划的大脑层需要：
1. **多模态输入**：视觉（深度图/点云）、IMU、当前状态
2. **高级决策**：DRL 或 NavFormer 架构
3. **输出**：导航轨迹 + 目标形变 Φ 的时间序列

---

## 四、对本科毕设的建议

### A. 关于大脑层：是否需要实现？

> [!IMPORTANT]
> **核心判断：作为本科毕设，你不需要完整实现大脑层来获得一篇优秀论文。**

**理由：**
1. 你的蓝图定位是一个**完整系统的架构设计**，脊髓和小脑已经体现了足够的工程与学术深度
2. 底层 GNN 连续拓扑流形 + 硬门控是一个**独立且完整的创新点**
3. 中层 MPC + Φ-aware PID 是另一个**独立可验证的贡献**
4. 把大脑层的 NavFormer 完整做出来，工作量等于一篇独立论文

**推荐策略：三选一**

| 方案 | 工作量 | 效果 | 推荐度 |
|------|--------|------|--------|
| **A. 规则大脑 (Scripted Brain)** | 🟢 1-2天 | Φ 时间序列的预定义任务 (如导航+变形穿越障碍) | ⭐⭐⭐⭐⭐ |
| **B. 简单 DRL (PPO/SAC)** | 🟡 1-2周 | 单一简单任务（如目标点导航）的端到端训练 | ⭐⭐⭐ |
| **C. NavFormer** | 🔴 1-2月+ | 完整 Transformer 序列决策 | ⭐ (不推荐本科阶段) |

**强烈推荐方案 A — 规则大脑 (Scripted Brain)：**

编写一个 `scripted_brain.py`，实现：
```python
class ScriptedBrain:
    def plan(self, scenario):
        """返回 (trajectory, phi_sequence) 
        例如: 
          1. O形态导航到门前 → 
          2. 变形为 1_shape 穿过窄门 → 
          3. 恢复 O形态继续导航
        """
```
这样你可以在论文中展示「**大脑给出的轨迹 + Φ 时间序列，经小脑 MPC 和脊髓 GNN 执行**」的完整流水线，演示视频效果极佳。

---

### B. 论文实验补充建议（优先级排序）

> [!IMPORTANT]
> 以下实验按优先级排序。前 4 个是**必做**，后面的是**加分项**。

#### 必做实验 (Must-Have)

##### 1. ⭐⭐⭐⭐⭐ GNN 消融实验 (Ablation Study)
**目的：** 证明 GNN + 硬门控的必要性

| 对比组 | 配置 | 指标 |
|--------|------|------|
| 完整 MorphGNN | GATv2 + Hard-Gating + 10D | ✅ Baseline |
| 无硬门控 | GATv2 + No Gating | MAE, 锁死关节误差 |
| MorphGNNLite | 2层 GATv2 + Hard-Gating | MAE, 推理延迟 |
| 纯 MLP (无 GNN) | 10D → MLP → 7 角度 | MAE, 泛化性 |
| 几何公式 | 手写分段线性 | MAE, 灵活性 |

##### 2. ⭐⭐⭐⭐⭐ MPC 轨迹跟踪定量评估
**目的：** 证明 MPC 小脑的跟踪精度

- **场景：** 直线、S 弯、8 字形、急转弯
- **指标：** 
  - Cross-Track Error (横向偏差)
  - Heading Error (航向偏差)
  - Settling Time (稳态时间)
- **对比组：** MPC vs PID-only vs 开环前馈
- **跨构型测试：** 在 Φ=0.0, 1.0, 1.5, 2.0, 3.0 五种形态下分别测试

##### 3. ⭐⭐⭐⭐⭐ Φ 连续变形全程演示
**目的：** 证明连续拓扑流形能让机器人平滑变形

- **场景：** Φ 从 0 连续扫到 3，记录每帧关节角度
- **图表：** Φ vs 7 关节角度曲线图（你已有 `phi_sweep.png` 的基础代码）
- **视频：** PyBullet 仿真视频，展示 8→O→门→1 全程平滑过渡

##### 4. ⭐⭐⭐⭐ 端到端演示：导航 + 变形穿障
**目的：** 证明三层架构的协同工作能力

- 使用「方案 A: 规则大脑」生成任务
- 场景：机器人以 O 形态导航 → 遇到窄缝 → 变形为 1 形态穿过 → 恢复 O 形态
- **这是答辩时最有冲击力的演示！**

---

#### 加分实验 (Nice-to-Have)

##### 5. ⭐⭐⭐ GNN 注意力权重可视化
- 利用 GATv2 的注意力系数，画出在不同 Φ 值下，节点间的注意力热力图
- 展示模型「学会了哪些关节之间的耦合关系」

##### 6. ⭐⭐⭐ PID Φ-aware 增益调度对比
- 对比固定增益 vs Φ-aware 增益调度在形变过程中的速度跟踪稳定性
- 画出误差曲线对比图

##### 7. ⭐⭐ 模型参数与推理延迟分析
- MorphGNN (全量) vs MorphGNNLite vs MLP 的参数量、FLOPs、单帧推理时间
- 证明 GNN 的推理延迟满足实时控制要求（<1ms）

##### 8. ⭐⭐ 边界行为分析
- 在 Φ=1.0 和 Φ=2.0 的构型切换边界处，GNN 的预测连续性
- 与几何公式在边界处的行为做对比

---

### C. 论文撰写结构建议

```
第1章 绪论
  1.1 可变拓扑机器人研究背景
  1.2 国内外研究现状 (变形机器人 + GNN + MPC)
  1.3 本文创新点与章节安排

第2章 相关理论基础
  2.1 图注意力网络 (GATv2)
  2.2 模型预测控制 (MPC)
  2.3 连续拓扑流形参数化

第3章 系统总体架构设计   ← 你的架构图放这里
  3.1 仿生三层控制架构 (大脑-小脑-脊髓)
  3.2 连续拓扑流形 Φ 的数学定义
  3.3 各层接口与数据流

第4章 底层脊髓：GNN 拓扑执行层  ← 核心创新章
  4.1 图结构建模 (7节点链式图)
  4.2 10D 节点特征设计
  4.3 硬门控机制
  4.4 训练策略 (边界加权 + 辅助损失)

第5章 中层小脑：运动学 MPC 控制层
  5.1 全向运动学模型
  5.2 MPC 代价函数设计
  5.3 PID 速度闭环 + Φ-aware 增益调度
  5.4 四轮舵运动学逆解

第6章 实验与分析         ← 上面列的实验放这里
  6.1 仿真平台搭建 (PyBullet)
  6.2 GNN 消融实验
  6.3 MPC 轨迹跟踪实验
  6.4 Φ 连续变形实验
  6.5 导航+穿障端到端演示

第7章 总结与展望
  7.1 工作总结
  7.2 未来工作 (NavFormer 大脑层)  ← 把 NavFormer 放在展望里
```

---

## 五、总结

你的项目已经建立了一个**工程质量极高**的仿真系统。底层 GNN 和中层 MPC 的完成度足以支撑一篇优秀的本科毕设论文。

**接下来的工作优先级：**

1. 🔴 **写一个 ScriptedBrain** (1-2天) → 完成端到端演示
2. 🔴 **跑 GNN 消融实验** (1天) → 核心创新的实验证据
3. 🟡 **跑 MPC 定量评估** (1天) → 中层的实验证据
4. 🟡 **录制 PyBullet 演示视频** → 答辩杀手锏
5. 🟢 **GNN 注意力可视化** → 论文亮点图

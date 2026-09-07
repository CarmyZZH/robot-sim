# 可变拓扑模块化机器人仿真与三层控制架构

基于 PyBullet 的可变拓扑机器人仿真系统，围绕“脊髓-小脑-大脑”三层仿生架构设计。底层用 GNN 将连续形变参数映射为关节角度，中层用运动学 MPC 做轨迹跟踪，上层用规则大脑做导航与变形决策。

```text
Benchmark Task / Scripted Brain
        ↓  参考轨迹 + 目标 Φ
Kinematic MPC
        ↓  世界系速度与舵轮执行指令
PyBullet Robot
        ↓
MorphGNN / Formula 生成 7 个变形关节角度
```

## 当前状态

| 层级 | 作用 | 状态 | 主要文件 |
| --- | --- | --- | --- |
| 底层脊髓 | 连续拓扑流形 Φ → 7 个变形关节角度 | 已完成 | `src/models/`, `src/utils/graph_converter.py` |
| 中层小脑 | MPC 轨迹跟踪与执行控制集成 | 已完成/整理中 | `src/controllers/`, `scripts/test_mpc_vs_pid_topology.py` |
| 上层大脑 | 场景感知、导航与变形决策 | 规则版已完成 | `src/brains/`, `src/envs/benchmark_env.py` |
| 仿真平台 | URDF 加载、物理参数、碰撞、IMU 式舵轮标定 | 已完成 | `src/envs/robot_env.py` |

## 核心设计

### 连续拓扑流形 Φ

机器人形态统一编码为连续参数 `Φ ∈ [0, 3]`，避免离散模态切换带来的不连续性。

| Φ 区间 | 对应形态 | 说明 |
| --- | --- | --- |
| `[0, 1)` | `8_shape` | 8 形态到边界形态 |
| `[1, 2]` | `O_shape` | O 形态连续压缩/展开 |
| `(2, 3)` | `door_shape` | 门形态到 1 形态 |
| `3` | `1_shape` | 直线形态，关节全部锁死 |

节点输入特征 10 维：

```text
[7D node one-hot | Φ_norm | is_locked | base_angle]
```

### GNN 硬门控

`MorphGNN` 使用 3 层 GATv2 预测关节角度。对物理锁死的关节，通过硬门控强制输出基准角度：

```text
Final_Output = Raw_GNN_Output * (1 - is_locked) + base_angle * is_locked
```

当前连续 Φ 版本 Val Loss 约为 `0.00216`，MAE 约为 `0.026 rad`（约 1.5°）。

### 世界坐标系运动控制

机器人形变时 base link 与四轮几何中心会发生偏移，因此运动学解算、速度反馈和 MPC 统一以世界坐标系为参考。

- `src/controllers/mpc_controller.py`：轻量级运动学 MPC，预测时域 N=10，SLSQP 求解。
- `scripts/test_mpc_vs_pid_topology.py`：包含 `PurePursuitPID`、MPC vs PID 拓扑对比和部分控制辅助逻辑。
- 各评估/GUI 脚本：承担 PyBullet 中的舵轮零位平滑、Φ 形态计算和控制链路集成。

### 基准场景与规则大脑

| 场景 | 配置 | 目标 |
| --- | --- | --- |
| Scene A | `BenchmarkTask(scene_id='A')` | 窄门穿越 |
| Scene B | `BenchmarkTask(scene_id='B')` | 多目标巡航 |
| Scene C | `BenchmarkTask(scene_id='C')` | 复合障碍赛道 |

`ScriptedBrain` 用有限状态机根据障碍类型和距离决定目标 Φ，向 MPC 输出局部参考轨迹。

## 目录结构

```text
robot-sim/
├── data/
│   ├── collected_datasets/       # GNN 训练/测试数据
│   └── my_robot/                 # URDF 与 STL 网格
├── experiments/                  # 模型权重、训练历史、评估图表
├── notebooks/                    # Jupyter 分析笔记本
├── scripts/                      # 数据采集、训练、评估和演示脚本
├── src/
│   ├── brains/                   # 上层大脑接口与实现
│   ├── controllers/              # 控制器模块，当前以 MPC 为主
│   ├── envs/                     # PyBullet 环境与 benchmark 任务
│   ├── models/                   # MorphGNN 与数据集封装
│   └── utils/                    # 图转换、坐标工具
├── midterm_report_and_plan.md    # 中期问题总结与后续计划
├── total_plan.md                 # 总体架构状态与实验建议
├── requirements.txt              # Python 依赖
└── README.md
```

## 安装

建议 Python 3.9，并使用 Conda 环境隔离依赖。

```bash
conda create -n robot-sim python=3.9 -y
conda activate robot-sim
pip install -r requirements.txt
```

如果 PyTorch / PyTorch Geometric 与本机 CUDA 版本不匹配，请按官方说明重新安装对应版本的 `torch`、`torchvision` 和 `torch-geometric`。

## 常用命令

### 生成 GNN 训练数据

```bash
python scripts/collect_gnn_data.py
```

输出位于 `data/collected_datasets/`，CSV 格式通常为：

```text
phi,target_j1,target_j2,target_j3,target_j4,target_j5,target_j6,target_j7
```

### 训练 MorphGNN

```bash
python scripts/train.py
```

结果写入 `experiments/phi_run_*`，包含 `best_model.pt`、`final_model.pt`、`config.json`、`history.json` 及训练曲线图。

### GUI 手动/半自动演示

```bash
python scripts/run_mpc_gui.py
```

集成 PyBullet GUI、GNN/公式切换、PID 开关和实时状态 HUD。

### 三层架构 benchmark 评估

```bash
python scripts/evaluate_brain.py
```

启动规则大脑、benchmark 场景、MPC 和控制执行全链路。

### 实验脚本

```bash
python scripts/eval_gnn_ablation.py          # GNN 消融实验
python scripts/eval_gnn_attention.py         # 注意力热力图
python scripts/eval_gnn_robustness.py        # 关节故障鲁棒性
python scripts/eval_mpc_benchmark.py         # MPC 轨迹跟踪评估
python scripts/eval_pid_ablation.py          # PID 增益调度消融
python scripts/test_mpc_pybullet.py          # MPC PyBullet 跟踪测试
python scripts/test_mpc_pure_kinematics.py   # 纯运动学 MPC 验证
python scripts/test_mpc_vs_pid_topology.py   # MPC vs PID 拓扑对比
python scripts/test_phi3_stability.py        # Φ=3 稳定性测试
python scripts/draw_ppt_diagrams.py          # 论文图表绘制
```

## 文件说明

### `src/envs/`

- `robot_env.py`：核心 PyBullet Gym 环境，负责 URDF 加载、物理参数、关节解析、IMU 式舵轮标定和动作执行。
- `configs.py`：O 形、8 形等构型定义和分步变形配置。
- `benchmark_env.py`：benchmark 场景包装器，创建窄门、平台、桥、圆柱等障碍。
- `task_manager.py`：A/B/C 场景轨迹生成、目标点和成功判定。

### `src/controllers/`

- `mpc_controller.py`：轻量级运动学 MPC，SLSQP 优化，预测时域 N=10。

### `src/models/`

- `model.py`：`MorphGNN` 与 `MorphGNNLite` 模型定义。
- `dataset.py`：CSV 数据转 PyTorch Geometric 图数据。

### `src/utils/`

- `graph_converter.py`：Φ 与形态/变形率映射、图特征矩阵构建、关节顺序转换。
- `coordinate_utils.py`：角度归一化等工具。

### `src/brains/`

- `base_brain.py`：上层大脑抽象接口和 MPC 轨迹切片工具。
- `scripted_brain.py`：基于有限状态机的规则大脑。

## 已知问题

- `src/` 顶层没有 `__init__.py`，当前脚本通过把 `src` 加入 `sys.path` 后导入子包，不影响运行。
- `experiments/` 存放大量模型权重和图片，通常不应提交到远程仓库。
- `.opencode/opencode.json` 中包含本地 OpenCode 配置和 API Key，公开项目前应移除或替换密钥。
- Φ 在 `Φ=2` 附近存在形态定义带来的快速切换行为，当前通过硬门控和边界加权训练缓解。

## 后续方向

- 使用 PPO/SAC 训练 DRL 大脑。
- 使用 Decision Transformer / NavFormer 做离线序列决策。
- 加入视觉或点云输入。
- 将 PyBullet 仿真迁移到真实模块化机器人硬件。

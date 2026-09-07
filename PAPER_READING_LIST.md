# 可变形机器人、形态表征与分层控制精读清单

## 1. 为什么读这些论文

本项目希望用统一的形态上下文 `Phi` 串联三层控制：

- 大脑根据环境和任务规划轨迹与目标形态；
- 小脑使用形态条件动力学和 MPC 跟踪轨迹；
- 脊髓把目标形态解码成关节动作并满足物理约束；
- 传感器估计实际形态并向上反馈，形成闭环。

当前 `phi in [0, 3]` 是一条人工设计形变路径上的一维坐标。长期目标是把它推广为：

`z_phi = Encoder(G, q_morph, lock_state, geometry, dynamics)`

因此需要同时理解模块化机器人、图网络、跨形态控制、潜在形态空间、学习型 MPC 和 Sim-to-Real。

## 2. 第一阶段：机器人与图学习基础

### 2.1 Modular Self-Reconfigurable Robot Systems

- 作者：Yim et al.
- 出处：IEEE Robotics & Automation Magazine, 2007
- 链接：https://ieeexplore.ieee.org/document/4141032/
- 核心：系统梳理模块化自重构机器人的机构、控制和主要挑战。
- 与项目关系：帮助严格区分“形态变化”和“拓扑重构”。如果连接关系没有断开重连，本项目应优先称为连续可变形或可重构机器人。
- 精读问题：本文的分类中，本项目属于哪一类？真正的拓扑变化需要满足哪些条件？

### 2.2 How Attentive Are Graph Attention Networks?

- 作者：Brody, Alon, Yahav
- 出处：ICLR 2022
- 链接：https://arxiv.org/abs/2105.14491
- 核心：指出原始 GAT 的静态注意力限制，并提出动态注意力 GATv2。
- 与项目关系：当前 MorphGNN 使用 GATv2，必须理解它相对 GCN、GAT 和 MLP 的真实优势。
- 精读问题：七节点固定链是否真的需要注意力？注意力权重能否解释物理耦合？

### 2.3 Graph Networks as Learnable Physics Engines for Inference and Control

- 作者：Sanchez-Gonzalez et al.
- 出处：ICML 2018
- 链接：https://proceedings.mlr.press/v80/sanchez-gonzalez18a.html
- 核心：用对象和关系图作为动力学归纳偏置，学习前向模型、系统辨识并支持轨迹优化。
- 与项目关系：提供了比“GNN拟合解析关节公式”更有说服力的方向，即学习不同形态下的残差动力学并放入 MPC。
- 精读问题：节点、边和全局特征分别应该表示什么？多步预测误差如何进入规划？

### 2.4 NerveNet: Learning Structured Policy with Graph Neural Networks

- 作者：Wang et al.
- 出处：ICLR 2018
- 链接：https://openreview.net/pdf?id=S1sqHMZCb
- 核心：按照机器人身体图构造策略网络，通过消息传递实现结构化控制和跨身体迁移。
- 与项目关系：是“身体图进入控制器”这条研究线的基础工作。
- 精读问题：参数共享带来的泛化来自哪里？固定单一结构时，GNN相对MLP还有多少优势？

## 3. 第二阶段：跨形态与通用控制器

### 3.1 Hardware Conditioned Policies for Multi-Robot Transfer Learning

- 作者：Chen, Murali, Gupta
- 出处：NeurIPS 2018
- 链接：https://proceedings.neurips.cc/paper/2018/hash/b8cfbf77a3d250a4523ba67a65a7d031-Abstract.html
- 核心：用显式硬件描述向量或学习得到的硬件嵌入条件化统一策略。
- 与项目关系：这是广义 `z_phi` 最直接的先例之一，证明“形态描述作为策略上下文”本身已有研究基础。
- 精读问题：显式描述和隐式嵌入各自适合哪些变化？如何测试未见形态的零样本迁移？

### 3.2 One Policy to Control Them All

- 作者：Huang, Mordatch, Pathak
- 出处：ICML 2020
- 链接：https://proceedings.mlr.press/v119/huang20d.html
- 核心：每个执行器使用共享模块网络并交换消息，使一个策略控制多种身体结构。
- 与项目关系：提供跨形态共享控制器的强 baseline。
- 精读问题：模块参数共享与全局形态向量条件化有什么差异？两者能否结合？

### 3.3 Learning Modular Robot Control Policies

- 作者：Whitman, Travers, Choset
- 出处：2021
- 链接：https://arxiv.org/abs/2105.10049
- 核心：从机器人设计图生成策略图，结合模型学习与轨迹优化，在未见设计和真实模块机器人上验证。
- 与项目关系：与本项目的图结构、模型控制和实机目标最直接，应作为重点逐图复现的论文。
- 精读问题：训练/测试设计怎样划分？真实机器人实验如何证明结构泛化而不是记忆？

### 3.4 My Body is a Cage

- 作者：Kurin et al.
- 出处：ICLR 2021
- 链接：https://openreview.net/forum?id=N3zUDGN5lO
- 核心：系统质疑形态图归纳偏置一定优于通用序列模型的假设，并展示 Transformer 的竞争力。
- 与项目关系：这是必须正面面对的反方工作。不能因为机器人能画成图，就默认 GNN 有必要。
- 精读问题：哪些实验会让固定七节点链上的 GNN 优势消失？你的 MLP、Transformer baseline 应怎样设计？

### 3.5 MetaMorph

- 作者：Gupta et al.
- 出处：ICLR 2022
- 链接：https://arxiv.org/abs/2203.11931
- 核心：把机器人形态作为 Transformer 的条件模态，预训练通用控制器并迁移到未见形态。
- 与项目关系：是 GNN 之外的重要强 baseline，也支持将形态看作上下文而非离散模式。
- 精读问题：形态 token 包含哪些信息？组合泛化实验如何设计？

### 3.6 AnyMorph

- 作者：Trabucco, Phielipp, Berseth
- 出处：ICML 2022
- 链接：https://proceedings.mlr.press/v162/trabucco22b.html
- 核心：不依赖人工形态描述，从控制目标中自动推断形态表示，并零样本迁移到新身体。
- 与项目关系：对应从手工标量 `phi` 走向学习型 `z_phi` 的路线。
- 精读问题：怎样防止潜变量只记住机器人 ID？如何证明它编码了控制相关的形态信息？

### 3.7 Universal Morphology Control via Contextual Modulation

- 作者：Xiong, Beck, Whiteson
- 出处：ICML 2023
- 链接：https://proceedings.mlr.press/v202/xiong23a.html
- 核心：使用形态上下文、超网络和形态依赖注意力生成不同机器人的控制参数。
- 与项目关系：概念上最接近“Phi作为跨层形态上下文”，需要明确本项目与它的差异。
- 主要差异机会：该类工作通常在一次任务中固定机器人身体；本项目强调同一实体在任务过程中连续变形。

### 3.8 Body Transformer

- 作者：Sferrazza et al.
- 出处：CoRL 2024 / PMLR 2025
- 链接：https://proceedings.mlr.press/v270/sferrazza25a.html
- 核心：将传感器和执行器组织为身体图，并用结构掩码指导 Transformer 注意力。
- 与项目关系：展示“图结构”不一定要求使用传统 GNN，也可作为注意力约束。
- 精读问题：图掩码与 GATv2 消息传递的归纳偏置有何不同？

## 4. 第三阶段：连续形态潜空间与联合优化

### 4.1 GLSO: Grammar-guided Latent Space Optimization

- 作者：Hu, Whitman, Choset
- 出处：CoRL 2022 / PMLR 2023
- 链接：https://proceedings.mlr.press/v205/hu23c.html
- 核心：用图 VAE 将离散、组合式机器人设计映射到低维连续潜空间，并在潜空间优化机器人设计。
- 与项目关系：为“复杂可量化Phi”提供直接方法论，但它主要解决设计搜索而不是在线分层控制。
- 精读问题：潜空间如何保证有效结构？一维Phi应如何推广到多维且保持可解释性？

### 4.2 Terrain-Aware Morphology Search for Self-Reconfigurable Robots

- 出处：Applied Soft Computing, 2026
- 链接：https://doi.org/10.1016/j.asoc.2025.114182
- 核心：用 Grammar VAE 把离散形态压缩到连续潜空间，再结合搜索算法和 MPPI 获得地形适应形态及行为。
- 与项目关系：这是目前最接近的竞争工作，已经包含连续形态潜空间、环境适应和预测控制。
- 可区分点：本项目应突出实时闭环形态估计、三层共享表示、连续变形过程控制和真实机器人验证。

### 4.3 An End-to-End Differentiable Framework for Contact-Aware Robot Design

- 作者：Xu et al.
- 出处：RSS 2021
- 链接：https://roboticsproceedings.org/rss17/p008.html
- 核心：通过可微仿真联合优化机器人形态和控制。
- 与项目关系：帮助理解形态和控制不是两个独立变量，而是耦合优化问题。
- 精读问题：本项目能否对轨迹和Phi联合优化，而不是由规则先选Phi再控制？

## 5. 第四阶段：学习型 MPC、在线估计与实机迁移

### 5.1 Learning-Based Model Predictive Control: Toward Safe Learning in Control

- 作者：Hewing et al.
- 出处：Annual Review of Control, Robotics, and Autonomous Systems, 2020
- 链接：https://doi.org/10.1146/annurev-control-090419-075625
- 核心：总结学习动力学、学习控制器设计和用 MPC 保证学习控制安全的三类路线。
- 与项目关系：是设计“形态条件残差动力学 + MPC”的理论入口。
- 精读问题：学习模型应该替代解析模型，还是只学习残差和不确定性？

### 5.2 RMA: Rapid Motor Adaptation for Legged Robots

- 作者：Kumar et al.
- 出处：RSS 2021
- 链接：https://roboticsproceedings.org/rss17/p011.html
- 核心：策略使用隐含环境参数，部署时由历史观测在线估计潜变量，实现快速实机适应。
- 与项目关系：可借鉴为 `phi_hat/z_phi` 估计器，避免把目标形态命令误当成真实形态。
- 精读问题：哪些传感器历史足以推断形态、负载、摩擦和执行器状态？

### 5.3 Sim-to-Real Transfer of Robotic Control with Dynamics Randomization

- 作者：Peng et al.
- 出处：ICRA 2018
- 链接：https://arxiv.org/abs/1710.06537
- 核心：训练时随机化动力学参数，使策略在真实机器人未知参数下保持鲁棒。
- 与项目关系：实机前应随机化质量、摩擦、电机延迟、关节间隙、传感器偏置和轮胎打滑。
- 精读问题：随机化范围如何由实机辨识确定，而不是任意设置？

## 6. 推荐阅读顺序

### 第1周：建立问题边界

1. Modular Self-Reconfigurable Robot Systems
2. How Attentive Are Graph Attention Networks?
3. Graph Networks as Learnable Physics Engines

### 第2周：理解图结构控制

1. NerveNet
2. Learning Modular Robot Control Policies
3. One Policy to Control Them All

### 第3周：理解形态表示与反方观点

1. Hardware Conditioned Policies
2. My Body is a Cage
3. MetaMorph
4. AnyMorph

### 第4周：寻找论文差异

1. Universal Morphology Control via Contextual Modulation
2. GLSO
3. Terrain-Aware Morphology Search
4. Body Transformer

### 第5周：准备实机和控制方法

1. Learning-Based MPC
2. RMA
3. Dynamics Randomization

## 7. 每篇精读统一回答的问题

1. 论文解决的科学问题是什么，而不是实现了什么系统？
2. 形态如何表示：ID、向量、图、token还是潜变量？
3. 形态在一个 episode 内是否变化？
4. 表示进入了策略、动力学模型、规划器还是全部层级？
5. 泛化按什么划分：随机样本、未见参数、未见形态还是未见拓扑？
6. 最强 baseline 是什么，是否公平？
7. 消融真正移除了哪个机制？
8. 是否有实机，实机测试次数和统计指标是什么？
9. 哪项假设可能不适用于本项目？
10. 这篇论文会如何审稿当前的 `Phi` 方法？

## 8. 读完后应形成的项目判断

`Phi` 不能只作为一个人为命名的控制量。论文贡献应落在以下至少两项：

1. 可解释且可扩展的形态上下文表示；
2. 同一形态状态贯穿大脑、小脑和脊髓的闭环接口；
3. 任务执行过程中连续形变，而不是每个 episode 固定一种机器人；
4. 形态条件动力学、约束或控制器，而不只是关节角解码；
5. 对未见形态区间、负载、摩擦和故障的泛化；
6. 真实机器人上的重复实验和统计显著性。

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


class RobotGraphConverter:
    # MorphGNN node order follows the physical chain from joint3 to joint7.
    MORPH_GRAPH_NODE_NAMES = (
        "joint3", "joint2", "joint1", "joint4", "joint5", "joint6", "joint7"
    )
    MORPH_GRAPH_TO_ENV = (2, 1, 0, 3, 4, 5, 6)

    def __init__(self):
        self.num_nodes = 11  # 7 Blocks + 4 Wheels
        self.x_dim = 12  # 特征维度固定为 12

        self.BLOCK_NODE_MAPPING = {
            0: "joint4", 1: "joint3", 2: "joint2", 3: "joint1",
            4: "joint5", 5: "joint6", 6: "joint7"
        }
        self.WHEEL_NODE_MAPPING = {
            7: ("joint8", "joint9", "joint10"),  # Pivot, Steer, Wheel
            8: ("joint11", "joint12", "joint13"),
            9: ("joint14", "joint15", "joint16"),
            10: ("joint17", "joint18", "joint19")
        }

        # 预计算图结构
        self.edge_index_11, self.edge_types_11 = self._build_edges()

        # Backward-compatible aliases for the legacy 11-node dynamics graph.
        # MorphGNN callers must use edge_index_7 explicitly.
        self.edge_index = self.edge_index_11
        self.edge_types = self.edge_types_11

    def _build_edges(self):
        edges = []
        types = []

        # Type 0: Block-Block (刚性连接)
        block_conns = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2],
                       [0, 4], [4, 0], [4, 5], [5, 4], [5, 6], [6, 5]]
        for s, d in block_conns:
            edges.append([s, d])
            types.append(0)

        # Type 1: Block-Wheel (关节连接)
        wheel_conns = [[3, 7], [7, 3], [1, 8], [8, 1], [4, 9], [9, 4], [6, 10], [10, 6]]
        for s, d in wheel_conns:
            edges.append([s, d])
            types.append(1)

        return (torch.tensor(edges, dtype=torch.long).t(),
                torch.tensor(types, dtype=torch.long))

    # ── 7 节点链式 GNN 推理用特征构建 ──────────────────────
    #  输入: phi (标量)
    #  输出: numpy [7, 10]，与 MorphJointDataset 训练格式一致
    #    [one-hot(7) | phi_norm(1) | is_locked(1) | base_angle(1)]

    @staticmethod
    def _get_lock_and_base(phi: float):
        """根据 Φ 返回 7 节点的 lock_status 和 base_angles (图节点顺序)"""
        import numpy as _np
        phi = _np.clip(phi, 0.0, 3.0)
        if phi < 1.0:
            lock   = [0, 1, 0, 1, 0, 1, 0]
            base   = [0.0, 1.57, 0.0, -1.57, 0.0, 1.57, 0.0]
        elif phi <= 2.0:
            lock   = [1, 0, 1, 0, 1, 0, 1]
            base   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif phi < 3.0:
            lock   = [0, 1, 0, 1, 0, 1, 0]
            base   = [0.0, -1.57, 0.0, 1.57, 0.0, -1.57, 0.0]
        else:
            lock   = [1, 1, 1, 1, 1, 1, 1]
            base   = [-1.57, -1.57, -1.57, 1.57, -1.57, -1.57, -1.57]
        return lock, base

    def get_feature_matrix(self, phi: float):
        """
        为 7 节点链式 GNN 推理构建特征矩阵。

        Returns:
            numpy.ndarray, shape [7, 10]
        """
        lock_status, base_angles = self._get_lock_and_base(phi)
        phi_norm = np.clip(phi, 0.0, 3.0) / 3.0
        features = np.zeros((7, 10), dtype=np.float32)
        for i in range(7):
            features[i, i] = 1.0            # one-hot 身份编码
            features[i, 7] = phi_norm        # 归一化 Φ
            features[i, 8] = lock_status[i]  # 锁定标志
            features[i, 9] = base_angles[i]  # 基准角度
        return features

    @classmethod
    def graph_to_env_angles(cls, graph_angles):
        """Convert seven MorphGNN outputs to joint1..joint7 environment order."""
        if len(graph_angles) != len(cls.MORPH_GRAPH_TO_ENV):
            raise ValueError(
                f"Expected 7 morphology joint angles, got {len(graph_angles)}"
            )
        env_angles = [0.0] * len(cls.MORPH_GRAPH_TO_ENV)
        for graph_idx, env_idx in enumerate(cls.MORPH_GRAPH_TO_ENV):
            env_angles[env_idx] = graph_angles[graph_idx]
        return env_angles

    # ── 7 节点链式边索引（与 MorphJointDataset 一致）──────
    @property
    def edge_index_7(self):
        """返回 7 节点链式图的边索引 [2, 12]"""
        edges = []
        for i in range(6):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        return torch.tensor(edges, dtype=torch.long).t()

    def world_to_body(self, raw_state):
        """将世界坐标系下的机身状态转换为局部坐标系"""
        quat = raw_state.get('base_orientation', [0, 0, 0, 1])
        lin_vel = raw_state.get('base_vel', [0, 0, 0])
        ang_vel = raw_state.get('base_ang_vel', [0, 0, 0])

        r = R.from_quat(quat)
        rpy = r.as_euler('xyz')  # Roll, Pitch, Yaw

        # 旋转矩阵的逆矩阵将向量从世界转到机身
        inv_r = r.inv()
        local_lin_vel = inv_r.apply(lin_vel)
        local_ang_vel = inv_r.apply(ang_vel)

        return {
            'roll': rpy[0],
            'pitch': rpy[1],
            'vx': local_lin_vel[0],
            'vy': local_lin_vel[1],
            'wz': local_ang_vel[2]  # Yaw rate
        }

    def get_feature_vector(self, raw_state, env_joint_map):
        """
        生成 Shape [11, 12] 的特征矩阵
        """
        js = raw_state['joint_states']
        block_lock_map = raw_state['block_lock_map']
        wheel_locks = raw_state['wheel_locks']

        # 计算机身局部状态
        body_state = self.world_to_body(raw_state)

        feats = []

        # === 1. Block Nodes (0-6) ===
        for i in range(7):
            j_name = self.BLOCK_NODE_MAPPING[i]
            p_id = env_joint_map[j_name]
            pos, vel = js[p_id]
            is_locked = block_lock_map.get(j_name, 0.0)

            # 基础特征: [TypeB, TypeW, Pos, Vel, AuxP, AuxV]
            f = [1.0, 0.0, pos, vel, 0.0, 0.0]

            # IMU 特征 (只在 Node 0 填充，其他为 0)
            if i == 0:
                imu = [body_state['roll'], body_state['pitch'],
                       body_state['vx'], body_state['vy'], body_state['wz']]
            else:
                imu = [0.0, 0.0, 0.0, 0.0, 0.0]

            # 组合: Base + IMU + Lock
            f.extend(imu)
            f.append(is_locked)
            feats.append(f)

        # === 2. Wheel Nodes (7-10) ===
        for i in range(4):
            idx = 7 + i
            # 映射: p=Pivot, s=Steer, w=Wheel
            p_n, s_n, w_n = self.WHEEL_NODE_MAPPING[idx]
            w_id, s_id, p_id = env_joint_map[w_n], env_joint_map[s_n], env_joint_map[p_n]

            _, w_vel = js[w_id]  # 驱动轮只看速度
            s_pos, s_vel = js[s_id]  # 转向轮看位置和速度
            p_pos, _ = js[p_id]  # 支点只看位置
            is_locked = wheel_locks[i]

            # Wheel 特征映射:
            # Main -> Steer (因为它是主要的控制自由度)
            # Aux Pos -> Pivot
            # Aux Vel -> Wheel Drive Vel
            # [TypeB, TypeW, SteerPos, SteerVel, PivotPos, DriveVel]
            f = [0.0, 1.0, s_pos, s_vel, p_pos, w_vel]

            # IMU (Wheel 节点不包含 IMU)
            f.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # Lock
            f.append(is_locked)
            feats.append(f)

        return np.array(feats, dtype=np.float32)

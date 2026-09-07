"""
GNN 训练数据集 (连续相位 Φ 版本)

CSV 格式:
  phi, target_j1, target_j2, target_j3, target_j4, target_j5, target_j6, target_j7

转换为 PyG Data:
  - x: 节点特征 [7, 10]
  - edge_index: 边索引 [2, 12] (双向链式)
  - y: 目标角度 [7, 1]
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Tuple, Optional


class MorphJointDataset(Dataset):
    """
    变形机器人关节角度数据集 (连续相位 Φ 版本)
    
    节点特征 (10 维):
      - 0-6: Node ID (one-hot)
      - 7:   Φ (归一化到 [0, 1])
      - 8:   Is_Locked
      - 9:   Base_Angle
    """
    
    # 图节点顺序: joint3, joint2, joint1, joint4, joint5, joint6, joint7
    ENV_TO_GRAPH_IDX = [2, 1, 0, 3, 4, 5, 6]
    
    def __init__(self, csv_path: str, transform=None, pre_transform=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        # 预计算边索引
        self.edge_index = self._build_edge_index()
        
        # 预处理所有数据
        self.data_list = self._preprocess_all()
        
        super(MorphJointDataset, self).__init__(None, transform, pre_transform)
        
    def _build_edge_index(self) -> torch.Tensor:
        """构建链式图的边索引 (双向)"""
        edges = []
        for i in range(6):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        return torch.tensor(edges, dtype=torch.long).t()
    
    def _get_positional_encoding(self, node_idx: int) -> List[float]:
        """获取节点位置编码 (7 维 one-hot)"""
        encoding = [0.0] * 7
        encoding[node_idx] = 1.0
        return encoding
    
    def _env_order_to_graph_order(self, env_angles: List[float]) -> List[float]:
        """将环境顺序角度转换为图节点顺序"""
        graph_angles = [0.0] * 7
        graph_angles[0] = env_angles[2]  # joint3
        graph_angles[1] = env_angles[1]  # joint2
        graph_angles[2] = env_angles[0]  # joint1
        graph_angles[3] = env_angles[3]  # joint4
        graph_angles[4] = env_angles[4]  # joint5
        graph_angles[5] = env_angles[5]  # joint6
        graph_angles[6] = env_angles[6]  # joint7
        return graph_angles
    
    @staticmethod
    def _get_lock_and_base(phi: float) -> tuple:
        """
        根据 Φ 获取 lock_status 和 base_angles (图节点顺序)
        
        与 graph_converter.py 中的逻辑保持一致
        """
        phi = np.clip(phi, 0.0, 3.0)
        
        if phi < 1.0:
            # 8_shape: J2, J4, J6 锁死
            lock_status = [0, 1, 0, 1, 0, 1, 0]
            base_angles = [0.0, 1.57, 0.0, -1.57, 0.0, 1.57, 0.0]
        elif phi <= 2.0:
            # O_shape: J1, J3, J5, J7 锁死
            lock_status = [1, 0, 1, 0, 1, 0, 1]
            base_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif phi < 3.0:
            # door_shape: J2, J4, J6 锁死
            lock_status = [0, 1, 0, 1, 0, 1, 0]
            base_angles = [0.0, -1.57, 0.0, 1.57, 0.0, -1.57, 0.0]
        else:
            # 1_shape: 所有关节锁死
            lock_status = [1, 1, 1, 1, 1, 1, 1]
            base_angles = [-1.57, -1.57, -1.57, 1.57, -1.57, -1.57, -1.57]
        
        return lock_status, base_angles
    
    def _preprocess_all(self) -> List[Data]:
        """预处理所有数据为 PyG Data 对象"""
        data_list = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            
            phi = float(row['phi'])
            
            # 环境顺序的目标角度
            env_angles = [
                row['target_j1'], row['target_j2'], row['target_j3'],
                row['target_j4'], row['target_j5'], row['target_j6'], row['target_j7']
            ]
            
            # 转换为图节点顺序
            graph_angles = self._env_order_to_graph_order(env_angles)
            
            # 获取锁定状态和基准角度
            lock_status, base_angles = self._get_lock_and_base(phi)
            
            # 锁死关节的目标值设为基准角度 (硬门控会强制输出基准角度)
            for i in range(7):
                if lock_status[i] == 1:
                    graph_angles[i] = base_angles[i]
            
            # 构建节点特征 [7, 10]
            node_features = []
            for i in range(7):
                pos_enc = self._get_positional_encoding(i)   # 7 维
                phi_norm = [phi / 3.0]                       # 1 维 (归一化)
                is_locked = [float(lock_status[i])]          # 1 维
                base_angle = [base_angles[i]]                # 1 维
                
                feat = pos_enc + phi_norm + is_locked + base_angle
                node_features.append(feat)
            
            x = torch.tensor(node_features, dtype=torch.float32)
            y = torch.tensor(graph_angles, dtype=torch.float32).unsqueeze(1)  # [7, 1]
            
            data = Data(
                x=x,
                edge_index=self.edge_index.clone(),
                y=y,
                phi=torch.tensor([phi], dtype=torch.float32)
            )
            data_list.append(data)
            
        return data_list
    
    def len(self) -> int:
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        return self.data_list[idx]


def load_dataset(csv_path: str, train_ratio: float = 0.8, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    加载数据集并划分训练/测试集
    """
    full_dataset = MorphJointDataset(csv_path)
    
    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train
    
    torch.manual_seed(seed)
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_test]
    )
    
    return train_dataset, test_dataset


def get_dataloader(dataset, batch_size: int = 32, shuffle: bool = True):
    """获取 DataLoader"""
    from torch_geometric.loader import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
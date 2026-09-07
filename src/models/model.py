"""
MorphGNN: 变形机器人关节角度预测模型 (连续相位 Φ 版本)

架构:
- 3 层 GATv2 (Graph Attention Network v2)
- 输入层: 10 -> 64
- 隐藏层: 64 -> 64
- 输出层: 64 -> 1
- 激活函数: ELU
- 注意力头数: 4

输入特征 (10 维):
- 0-6: Node One-Hot (Positional Encoding)
- 7:   Φ (全局相位, 归一化到 [0, 1])
- 8:   Is_Locked (1.0=锁死, 0.0=可动)
- 9:   Base_Angle (锁死关节的基准角度)

核心机制: 硬门控 (Hard Gating) - 基于基准角度
- Final_Output = Raw_GNN_Output × (1.0 - Is_Locked) + Base_Angle × Is_Locked
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch


class MorphGNN(nn.Module):
    """
    变形机器人 GNN 模型 (连续相位 Φ 版本)
    
    输入特征 (10 维):
    - 0-6: Node One-Hot (Positional Encoding)
    - 7:   Φ (全局相位, 归一化 [0, 1])
    - 8:   Is_Locked (1.0=锁死, 0.0=可动)
    - 9:   Base_Angle (锁死关节的基准角度)
    
    输出:
    - 每个节点的目标角度 (弧度)
    """
    
    def __init__(self, 
                 in_channels: int = 10, 
                 hidden_channels: int = 64, 
                 out_channels: int = 1,
                 heads: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            in_channels: 输入特征维度 (10)
            hidden_channels: 隐藏层维度 (64)
            out_channels: 输出维度 (1, 角度)
            heads: 注意力头数 (4)
            dropout: Dropout 概率
        """
        super(MorphGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # ============================================
        # GATv2 层
        # ============================================
        
        # 输入层: 10 -> 64 (4 heads, 每个 head 输出 16)
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # 隐藏层 1: 64 -> 64
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # 隐藏层 2: 64 -> 64
        self.conv3 = GATv2Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # 输出层: 64 -> 1 (MLP)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, 10]
            edge_index: 边索引 [2, E]
        
        Returns:
            输出角度 [N, 1] (已应用硬门控)
        """
        # 提取 Is_Locked 和 Base_Angle 特征 (用于硬门控)
        is_locked = x[:, 8:9]    # [N, 1]
        base_angle = x[:, 9:10]  # [N, 1]
        
        # ============================================
        # GATv2 消息传递
        # ============================================
        
        # Layer 1
        out = self.conv1(x, edge_index)
        h = out[0] if isinstance(out, tuple) else out
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Layer 2
        out = self.conv2(h, edge_index)
        h = out[0] if isinstance(out, tuple) else out
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Layer 3
        out = self.conv3(h, edge_index)
        h = out[0] if isinstance(out, tuple) else out
        h = self.bn3(h)
        h = F.elu(h)
        
        # ============================================
        # 输出层 + 硬门控
        # ============================================
        
        # MLP 输出
        raw_output = self.out_mlp(h)  # [N, 1]
        
        # 硬门控: Final = Raw × (1 - Is_Locked) + Base_Angle × Is_Locked
        final_output = raw_output * (1.0 - is_locked) + base_angle * is_locked
        
        # 始终返回双输出 (final, raw)，以兼容消融评估等需要 raw_output 的场景
        return final_output, raw_output
    
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return hard-gated joint angles in inference mode."""
        self.eval()
        with torch.no_grad():
            final_output, _ = self.forward(x, edge_index)
        return final_output


class MorphGNNLite(nn.Module):
    """
    轻量版 MorphGNN (连续相位 Φ 版本)
    2 层 GATv2, 更少的隐藏维度
    """
    
    def __init__(self, 
                 in_channels: int = 10, 
                 hidden_channels: int = 32, 
                 out_channels: int = 1,
                 heads: int = 2):
        super(MorphGNNLite, self).__init__()
        
        self.conv1 = GATv2Conv(in_channels, hidden_channels // heads, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, concat=True)
        self.out_linear = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        is_locked = x[:, 8:9]
        base_angle = x[:, 9:10]
        
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        raw_output = self.out_linear(h)
        
        # 硬门控
        final_output = raw_output * (1.0 - is_locked) + base_angle * is_locked
        
        return final_output, raw_output

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return hard-gated joint angles in inference mode."""
        self.eval()
        with torch.no_grad():
            final_output, _ = self.forward(x, edge_index)
        return final_output


def create_model(model_type: str = "full", device: str = "cuda") -> nn.Module:
    """
    创建模型实例
    
    Args:
        model_type: "full" 或 "lite"
        device: "cuda" 或 "cpu"
    """
    if model_type == "full":
        model = MorphGNN()
    else:
        model = MorphGNNLite()
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

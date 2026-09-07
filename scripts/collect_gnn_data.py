"""
GNN 训练数据收集脚本 (连续相位 Φ 版本)

全局变形相位 Φ ∈ [0, 3]:
  Φ ∈ [0, 1): 8 形态,   rate = Φ
  Φ ∈ [1, 2]: O 形态,   rate = 2 - Φ
  Φ ∈ (2, 3): 门 形态,  rate = Φ - 2
  Φ = 3:      1 形态 (固定)

数据格式:
  phi, target_j1, target_j2, target_j3, target_j4, target_j5, target_j6, target_j7
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------
# 路径设置
# ---------------------------------------------------------
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, os.path.join(project_root, 'src'))


# =========================================================
# 几何公式
# =========================================================

def compute_o_shape_block_angles(flatten_rate: float) -> list:
    """
    O 形态: 根据扁平率计算关节角度
    
    J2/4/6 关于 rate=0.5 (Φ=1.5, 初始圆形 O) 对称变化:
    - flatten_rate = 1 (Φ=1): J4=-1.57, J2=1.57, J6=1.57 → 连接 8_shape
    - flatten_rate = 0.5 (Φ=1.5): 全部 0 → 初始圆形 O (真正的零点)
    - flatten_rate = 0 (Φ=2): J4=1.57, J2=-1.57, J6=-1.57 → 连接 door_shape
    """
    flatten_rate = np.clip(flatten_rate, 0.0, 1.0)
    
    # 关键: 以 rate=0.5 为对称中心, J2/4/6 线性穿过零点
    joint4_angle = -1.57 * (2.0 * flatten_rate - 1.0)
    joint2_angle = -joint4_angle
    joint6_angle = -joint4_angle
    
    return [
        0.0,            # joint1
        joint2_angle,   # joint2
        0.0,            # joint3
        joint4_angle,   # joint4
        0.0,            # joint5
        joint6_angle,   # joint6
        0.0,            # joint7
    ]


def compute_8_shape_block_angles(lift_rate: float) -> list:
    """
    8 形态: 根据抬起率计算关节角度
    
    - lift_rate = 0: 完全 8 形态 (J1/3/5/7 = 1.57)
    - lift_rate = 1: 抬起状态 (= O 形态扁平状态)
    """
    lift_rate = np.clip(lift_rate, 0.0, 1.0)
    
    corner_angle = 1.57 * (1.0 - lift_rate)
    
    return [
        corner_angle,   # joint1
        1.57,           # joint2 - 固定
        corner_angle,   # joint3
        -1.57,          # joint4 - 固定
        corner_angle,   # joint5
        1.57,           # joint6 - 固定
        corner_angle,   # joint7
    ]


def compute_door_shape_block_angles(rate: float) -> list:
    """
    门 形态: 根据变形率计算关节角度
    
    - rate = 0: 门形态起始 (J2=-1.57, J4=1.57, J6=-1.57, 其余=0)
    - rate = 1: = 1 形态 (J1/3/5/7=-1.57, J2/6=-1.57, J4=1.57)
    
    锁死关节: J2=-1.57, J4=1.57, J6=-1.57 (固定不变)
    可动关节: J1, J3, J5, J7 从 0 线性减小到 -1.57
    """
    rate = np.clip(rate, 0.0, 1.0)
    
    # J1, J3, J5, J7: 0 → -1.57 (随 rate 线性)
    corner_angle = -1.57 * rate
    
    return [
        corner_angle,   # joint1: 0 → -1.57
        -1.57,          # joint2: 锁死
        corner_angle,   # joint3: 0 → -1.57
        1.57,           # joint4: 锁死
        corner_angle,   # joint5: 0 → -1.57
        -1.57,          # joint6: 锁死
        corner_angle,   # joint7: 0 → -1.57
    ]


def get_1_shape_angles() -> list:
    """
    1 形态: 固定的关节角度 (所有关节锁死)
    
    J1=-1.57, J2=-1.57, J3=-1.57, J4=1.57, J5=-1.57, J6=-1.57, J7=-1.57
    """
    return [-1.57, -1.57, -1.57, 1.57, -1.57, -1.57, -1.57]


# =========================================================
# 统一 Φ 映射
# =========================================================

def compute_angles_from_phi(phi: float) -> list:
    """
    根据全局相位 Φ 计算 7 个关节的目标角度 (环境顺序)
    
    Args:
        phi: 全局相位 [0, 3]
        
    Returns:
        list: [j1, j2, j3, j4, j5, j6, j7] 目标角度
    """
    phi = np.clip(phi, 0.0, 3.0)
    
    if phi <= 1.0:
        # 8_shape: rate = phi
        return compute_8_shape_block_angles(phi)
    elif phi <= 2.0:
        # O_shape: rate = 2 - phi
        return compute_o_shape_block_angles(2.0 - phi)
    elif phi < 3.0:
        # door_shape: rate = phi - 2
        return compute_door_shape_block_angles(phi - 2.0)
    else:
        # 1_shape: 固定状态
        return get_1_shape_angles()


# =========================================================
# 数据收集
# =========================================================

def collect_data(num_samples_per_segment: int = 1001) -> pd.DataFrame:
    """
    收集全形态训练数据
    
    采样策略:
      - Φ ∈ [0, 1]: 8 形态,   num_samples_per_segment 个点
      - Φ ∈ [1, 2]: O 形态,   num_samples_per_segment 个点
      - Φ ∈ [2, 3]: 门 形态,  num_samples_per_segment 个点
      - Φ = 3:      1 形态,   1 个点 (固定状态)
      
      注意: 边界点 Φ=1, Φ=2 会被两个区间各采集一次，
      但由于物理状态完全相同，这是期望的 (增强边界学习)
    
    Args:
        num_samples_per_segment: 每个形态区间的采样点数 (默认 1001)
    
    Returns:
        pd.DataFrame
    """
    data = []
    
    print(f"[Data Collection] 开始收集全形态数据...")
    print(f"  每段采样点数: {num_samples_per_segment}")
    
    # ---------------------------------------------------------
    # 区间 A: 8 形态 (Φ ∈ [0, 1])
    # ---------------------------------------------------------
    print(f"\n[Φ ∈ [0, 1]] 收集 8 形态数据...")
    phis_8 = np.linspace(0.0, 1.0, num_samples_per_segment)
    for phi in phis_8:
        angles = compute_angles_from_phi(phi)
        data.append(_make_row(phi, angles))
    print(f"  ✓ 收集完成: {len(phis_8)} 条")
    
    # ---------------------------------------------------------
    # 区间 B: O 形态 (Φ ∈ [1, 2])
    # ---------------------------------------------------------
    print(f"\n[Φ ∈ [1, 2]] 收集 O 形态数据...")
    phis_o = np.linspace(1.0, 2.0, num_samples_per_segment)
    for phi in phis_o:
        angles = compute_angles_from_phi(phi)
        data.append(_make_row(phi, angles))
    print(f"  ✓ 收集完成: {len(phis_o)} 条")
    
    # ---------------------------------------------------------
    # 区间 C: 门 形态 (Φ ∈ [2, 3])
    # ---------------------------------------------------------
    print(f"\n[Φ ∈ [2, 3]] 收集 门 形态数据...")
    phis_door = np.linspace(2.0, 3.0, num_samples_per_segment)
    for phi in phis_door:
        angles = compute_angles_from_phi(phi)
        data.append(_make_row(phi, angles))
    print(f"  ✓ 收集完成: {len(phis_door)} 条")
    
    # ---------------------------------------------------------
    # 边界密集采样 (Φ = 0, 1, 2, 3 各 ±0.05 窗口)
    # ---------------------------------------------------------
    print(f"\n[边界增强] 注入边界密集采样...")
    boundary_phis = [0.0, 1.0, 2.0, 3.0]
    boundary_count = 0
    for bp in boundary_phis:
        # 精确边界点重复 10 次
        for _ in range(10):
            angles = compute_angles_from_phi(bp)
            data.append(_make_row(bp, angles))
            boundary_count += 1
        # ±0.05 密集带 (51 个点)
        for delta in np.linspace(-0.05, 0.05, 51):
            phi_dense = np.clip(bp + delta, 0.0, 3.0)
            angles = compute_angles_from_phi(phi_dense)
            data.append(_make_row(phi_dense, angles))
            boundary_count += 1
    print(f"  ✓ 注入边界样本: {boundary_count} 条")
    
    df = pd.DataFrame(data)
    print(f"\n[Data Collection] 总计收集: {len(df)} 条数据")
    
    return df


def _make_row(phi: float, angles: list) -> dict:
    """构造一行数据"""
    return {
        'phi': round(phi, 6),
        'target_j1': round(angles[0], 6),
        'target_j2': round(angles[1], 6),
        'target_j3': round(angles[2], 6),
        'target_j4': round(angles[3], 6),
        'target_j5': round(angles[4], 6),
        'target_j6': round(angles[5], 6),
        'target_j7': round(angles[6], 6),
    }


def save_data(df: pd.DataFrame, output_dir: str) -> str:
    """保存数据到 CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gnn_phi_data_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"\n[Save] 数据已保存到: {filepath}")
    
    return filepath


def main():
    print("=" * 60)
    print("   GNN 训练数据收集 (连续相位 Φ 版本)")
    print("=" * 60)
    
    NUM_SAMPLES_PER_SEGMENT = 1001
    
    # 收集数据
    df = collect_data(num_samples_per_segment=NUM_SAMPLES_PER_SEGMENT)
    
    # 输出目录
    output_dir = os.path.join(project_root, "data", "collected_datasets")
    
    # 保存
    filepath = save_data(df, output_dir)
    
    # 数据预览
    print("\n[Preview] 数据预览:")
    print(df.head(10).to_string(index=False))
    print("...")
    print(df.tail(5).to_string(index=False))
    
    # 统计信息
    print("\n[Stats] 数据统计:")
    n_8 = len(df[df['phi'] <= 1.0])
    n_o = len(df[(df['phi'] > 1.0) & (df['phi'] <= 2.0)])
    n_door = len(df[df['phi'] > 2.0])
    print(f"  8 形态 (Φ ∈ [0,1]):   {n_8} 条")
    print(f"  O 形态 (Φ ∈ (1,2]):   {n_o} 条")
    print(f"  门/1 形态 (Φ ∈ (2,3]): {n_door} 条")
    print(f"  总计: {len(df)} 条")
    
    # 边界一致性检查
    print("\n[Check] 边界一致性验证:")
    angles_8_at_1 = compute_angles_from_phi(1.0)
    angles_o_at_1 = compute_angles_from_phi(1.0)
    print(f"  Φ=1 (8→O 边界): {[f'{a:.2f}' for a in angles_8_at_1]}")
    
    angles_o_at_2 = compute_angles_from_phi(2.0)
    angles_door_at_2 = compute_angles_from_phi(2.0)
    print(f"  Φ=2 (O→门 边界): {[f'{a:.2f}' for a in angles_o_at_2]}")
    
    angles_door_at_3 = compute_angles_from_phi(3.0)
    angles_1 = get_1_shape_angles()
    print(f"  Φ=3 (门→1 边界): {[f'{a:.2f}' for a in angles_door_at_3]}")
    print(f"  1_shape 固定角度:  {[f'{a:.2f}' for a in angles_1]}")
    
    print("\n" + "=" * 60)
    print("   数据收集完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

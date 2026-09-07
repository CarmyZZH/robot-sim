"""
GNN 门控机制消融实验 (GNN Ablation Study)

评估不同模型配置在连续 Phi 测试集上的预测误差（MAE）:
1. GNN w/ Hard-Gating (Ours - Final Output)
2. GNN w/o Gating (Raw Output)
3. Formula Baseline (纯几何计算基准)

生成柱状图对比并保存到 experiments/gnn_ablation_{timestamp}/
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------
# 路径设置
# ---------------------------------------------------------
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, os.path.join(project_root, 'src'))

from models.model import MorphGNN, MorphGNNLite
from models.dataset import MorphJointDataset

# ── 兜底：如果 CSV 数据文件丢失，自动按公式生成测试集 ──
import tempfile, csv

def generate_synthetic_csv():
    """在内存中生成与训练时一致的 Φ 测试数据集（4000+ 样本）"""
    fd, tmp_path = tempfile.mkstemp(suffix='.csv', prefix='gnn_ablation_')
    with os.fdopen(fd, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['phi','target_j1','target_j2','target_j3','target_j4','target_j5','target_j6','target_j7'])
        # 基础采样: Φ ∈ [0, 3], 步长 0.001
        for v in np.arange(0, 3.001, 0.001):
            angles = formula_angles(v)
            writer.writerow([round(v, 6)] + [round(a, 6) for a in angles])
        # 边界加密: Φ=1.0 和 Φ=2.0 附近
        for delta in np.arange(-0.05, 0.051, 0.001):
            for center in [1.0, 2.0]:
                v = center + delta
                if v < 0 or v > 3:
                    continue
                angles = formula_angles(v)
                writer.writerow([round(v, 6)] + [round(a, 6) for a in angles])
    print(f"[Data] CSV 文件缺失，已自动按公式生成临时数据集: {os.path.basename(tmp_path)}")
    return tmp_path

def formula_angles(phi):
    """几何公式基准 (与环境中的逻辑保持一致)"""
    phi = np.clip(phi, 0.0, 3.0)
    if phi <= 1.0:
        r = phi
        ca = 1.57 * (1.0 - r)
        return [ca, 1.57, ca, -1.57, ca, 1.57, ca]
    elif phi <= 2.0:
        r = 2.0 - phi
        j4 = -1.57 * (2.0 * r - 1.0)
        return [0.0, -j4, 0.0, j4, 0.0, -j4, 0.0]
    elif phi < 3.0:
        r = phi - 2.0
        ca = -1.57 * r
        return [ca, -1.57, ca, 1.57, ca, -1.57, ca]
    else:
        return [-1.57, -1.57, -1.57, 1.57, -1.57, -1.57, -1.57]

def evaluate_ablation(model, loader, device):
    """在测试集上评估三种方法"""
    model.eval()
    
    all_targets = []
    all_gnn_final = []
    all_gnn_raw = []
    all_formula = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # GNN 前向 (现在 eval 模式也返回双输出)
            final_out, raw_out = model(batch.x, batch.edge_index)
            
            all_targets.append(batch.y.cpu().numpy())
            all_gnn_final.append(final_out.cpu().numpy())
            all_gnn_raw.append(raw_out.cpu().numpy())
            
            # Formula 计算
            phis = batch.phi.cpu().numpy()
            formula_batch = []
            for i in range(batch.num_graphs):
                phi = phis[i] if phis.ndim == 1 else phis[i][0]
                env_angles = formula_angles(phi)
                # 转换回图节点顺序 (0:R_H_R, 1:R_H_L, 2:F_H_L, 3:F_H_R, 4:R_V_R, 5:R_V_L, 6:F_V_L)
                graph_angles = [env_angles[2], env_angles[1], env_angles[0], 
                                env_angles[3], env_angles[4], env_angles[5], env_angles[6]]
                formula_batch.extend(graph_angles)
            
            all_formula.append(np.array(formula_batch).reshape(-1, 1))

    # 恢复 model 状态
    model.eval()

    targets = np.concatenate(all_targets, axis=0)
    gnn_final = np.concatenate(all_gnn_final, axis=0)
    gnn_raw = np.concatenate(all_gnn_raw, axis=0)
    formula = np.concatenate(all_formula, axis=0)
    
    # 计算 MAE
    mae_final = np.mean(np.abs(gnn_final - targets))
    mae_raw = np.mean(np.abs(gnn_raw - targets))
    mae_formula = np.mean(np.abs(formula - targets))
    
    return {
        'GNN w/ Gating (Ours)': mae_final,
        'GNN w/o Gating (Raw)': mae_raw,
        'Formula Baseline': mae_formula
    }

def main():
    parser = argparse.ArgumentParser(description='GNN Ablation Study')
    parser.add_argument('--model_dir', type=str, default=None, help='Path to exp dir containing best_model.pt')
    parser.add_argument('--data', type=str, default=None, help='Path to test dataset CSV')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
    print("=" * 60)
    print("   GNN Ablation Study: Hard-Gating Mechanism")
    print("=" * 60)
    
    # 自动寻找最近的模型和数据
    if args.model_dir is None:
        exp_base = os.path.join(project_root, "experiments")
        runs = [d for d in os.listdir(exp_base) if d.startswith('phi_run_')]
        if not runs:
            print("[FAIL] No trained model found.")
            return
        runs.sort(reverse=True)
        model_path = os.path.join(exp_base, runs[0], 'best_model.pt')
        print(f"[Model] Using latest checkpoint: {runs[0]}/best_model.pt")
    else:
        model_path = os.path.join(args.model_dir, 'best_model.pt')
        
    if args.data is None:
        data_dir = os.path.join(project_root, "data", "collected_datasets")
        os.makedirs(data_dir, exist_ok=True)
        csv_files = [f for f in os.listdir(data_dir) if f.startswith('gnn_phi_data') and f.endswith('.csv')]
        if not csv_files:
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if csv_files:
            csv_files.sort(reverse=True)
            data_path = os.path.join(data_dir, csv_files[0])
            print(f"[Data] Using dataset: {os.path.basename(data_path)}")
        else:
            data_path = generate_synthetic_csv()
    else:
        data_path = args.data
        
    # 加载数据
    dataset = MorphJointDataset(data_path)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    # 加载模型
    model = MorphGNN().to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[OK] Model loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
        return
        
    # 运行评估
    print("\nRunning ablation evaluation...")
    results = evaluate_ablation(model, loader, device)
    
    print("\n--- Ablation Results (MAE in rad) ---")
    for name, mae in results.items():
        print(f"{name:25s} : {mae:.4f}")
        
    # 可视化输出
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(project_root, 'experiments', f'gnn_ablation_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    names = list(results.keys())
    values = list(results.values())
    colors = ['#E91E63', '#9C27B0', '#607D8B']
    
    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Mean Absolute Error (rad)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Hard-Gating Mechanism in MorphGNN', fontsize=14, fontweight='bold')
    
    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    img_path = os.path.join(out_dir, 'ablation_mae_bars.png')
    plt.savefig(img_path, dpi=200)
    plt.close()
    
    print(f"\n[OK] Ablation chart saved to: {out_dir}/")

if __name__ == '__main__':
    main()

"""
MorphGNN 训练脚本 (连续相位 Φ 版本)

训练 GNN 模型预测机器人关节角度
CSV 数据格式: phi, target_j1, ..., target_j7
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------
# 路径设置
# ---------------------------------------------------------
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, os.path.join(project_root, 'src'))

from models.model import MorphGNN, MorphGNNLite, count_parameters
from models.dataset import MorphJointDataset, load_dataset


def train_epoch(model, loader, optimizer, criterion, device, aux_weight=0.5):
    """训练一个 epoch (辅助损失 + 边界加权)"""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 双输出: (硬门控输出, 原始网络输出)
        final_out, raw_out = model(batch.x, batch.edge_index)
        
        # ---- 边界加权 ----
        # 从节点特征中反归一化取回 Φ (第 7 维, 归一化到 [0,1])
        phi_val = batch.x[:, 7] * 3.0  # [N]
        
        # 到最近整数边界 (0, 1, 2, 3) 的距离
        boundary_points = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
        dist_to_boundary = torch.abs(phi_val.unsqueeze(1) - boundary_points.unsqueeze(0))  # [N, 4]
        min_dist = dist_to_boundary.min(dim=1).values  # [N]
        
        # 指数衰减权重: 边界附近权重高, 远离边界权重→1
        boundary_weight = 1.0 + 3.0 * torch.exp(-min_dist / 0.05)  # [N]
        boundary_weight = boundary_weight.unsqueeze(1)  # [N, 1]
        
        # ---- 主损失 (带边界加权) ----
        loss_main_per_node = (final_out - batch.y) ** 2 * boundary_weight  # [N, 1]
        loss_main = loss_main_per_node.mean()
        
        # ---- 辅助损失: 让 raw_output 即便在锁死期也逼近目标 ----
        loss_aux = criterion(raw_out, batch.y)
        
        # ---- 总损失 ----
        loss = loss_main + aux_weight * loss_aux
        
        loss.backward()
        optimizer.step()
        
        # 记录主损失 (评估实际门控效果)
        total_loss += loss_main.item() * batch.num_graphs
        num_samples += batch.num_graphs
    
    return total_loss / num_samples


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_samples = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _ = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs
            
            all_preds.append(out.cpu())
            all_targets.append(batch.y.cpu())
    
    avg_loss = total_loss / num_samples
    
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    mae = torch.mean(torch.abs(preds - targets)).item()
    
    return avg_loss, mae


def plot_training_dashboard(train_losses, val_losses, lr_history, mae_history, 
                            best_epoch, save_path):
    """绘制训练仪表盘 (2×2 多面板)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MorphGNN Training Dashboard (Continuous Φ)', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(train_losses) + 1)
    
    # ---- Panel 1: Loss Curves (log scale) ----
    ax = axes[0, 0]
    ax.semilogy(epochs, train_losses, label='Train Loss', linewidth=2, color='#2196F3')
    ax.semilogy(epochs, val_losses, label='Val Loss', linewidth=2, color='#FF5722')
    ax.axvline(x=best_epoch + 1, color='#4CAF50', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch+1})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (log)')
    ax.set_title('Loss Curves')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ---- Panel 2: Learning Rate Schedule ----
    ax = axes[0, 1]
    ax.plot(epochs[:len(lr_history)], lr_history, linewidth=2, color='#9C27B0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('LR Schedule')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    
    # ---- Panel 3: MAE per Epoch ----
    ax = axes[1, 0]
    ax.plot(epochs[:len(mae_history)], mae_history, linewidth=2, color='#FF9800')
    ax.axhline(y=0.01, color='#4CAF50', linestyle=':', alpha=0.5, label='0.01 rad target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val MAE (rad)')
    ax.set_title('Validation MAE')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ---- Panel 4: Final Loss Ratio ----
    ax = axes[1, 1]
    if len(train_losses) > 10:
        ratio = [v / t if t > 0 else 1 for t, v in zip(train_losses, val_losses)]
        ax.plot(epochs, ratio, linewidth=2, color='#607D8B')
        ax.axhline(y=1.0, color='#4CAF50', linestyle='--', alpha=0.5, label='No overfit')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val/Train Loss Ratio')
        ax.set_title('Overfit Monitor')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(3, max(ratio[:min(len(ratio), 50)])))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



def plot_prediction_comparison(model, test_loader, device, save_path):
    """预测对比图 (按 Φ 区间着色)"""
    model.eval()
    all_preds = []
    all_targets = []
    all_phis = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out, _ = model(batch.x, batch.edge_index)
            all_preds.append(out.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            all_phis.append(batch.phi.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    phis = np.concatenate(all_phis, axis=0)
    
    colors = []
    for phi_val in phis:
        phi = phi_val[0] if hasattr(phi_val, '__len__') else phi_val
        if phi <= 1.0:
            c = '#2196F3'     # blue - 8 形态
        elif phi <= 2.0:
            c = '#4CAF50'     # green - O 形态
        else:
            c = '#FF9800'     # orange - 门/1 形态
        colors.extend([c] * 7)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Prediction Quality Analysis', fontsize=16, fontweight='bold')
    
    # Panel 1: Scatter
    ax = axes[0]
    for c, label in [('#2196F3', '8_shape'), ('#4CAF50', 'O_shape'), ('#FF9800', 'door/1')]:
        mask = [i for i, x in enumerate(colors) if x == c]
        if mask:
            ax.scatter(np.array(targets)[mask], np.array(preds)[mask], 
                      alpha=0.5, s=12, c=c, label=label, edgecolors='none')
    ax.plot([-2, 2], [-2, 2], 'r--', linewidth=2, alpha=0.7, label='Perfect')
    ax.set_xlabel('Target Angle (rad)', fontsize=12)
    ax.set_ylabel('Predicted Angle (rad)', fontsize=12)
    ax.set_title('Predicted vs Target')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    
    # Panel 2: Error histogram
    ax = axes[1]
    errors = preds - targets
    ax.hist(errors, bins=80, color='#607D8B', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Prediction Error (rad)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Error Distribution (std={np.std(errors):.4f})')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_phi_sweep(model, device, save_path):
    """
    Φ 扫描图: 对比 GNN 预测 vs 公式在 Φ ∈ [0, 3] 上的输出
    """
    from utils.graph_converter import RobotGraphConverter
    converter = RobotGraphConverter()
    edge_index = converter.edge_index_7.to(device)
    
    # 几何公式 (内联)
    def formula_angles(phi):
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
    
    phis = np.linspace(0, 3, 301)
    formula_all = np.array([formula_angles(p) for p in phis])  # [301, 7]
    
    # GNN 预测
    model.eval()
    gnn_all = []
    with torch.no_grad():
        for phi in phis:
            x = torch.tensor(converter.get_feature_matrix(phi), dtype=torch.float32).to(device)
            pred = model.predict(x, edge_index).squeeze(-1).cpu().numpy()
            env = converter.graph_to_env_angles(pred)
            gnn_all.append(env)
    gnn_all = np.array(gnn_all)  # [301, 7]
    
    # 绘制 7 关节的对比
    joint_names = [f'J{i+1}' for i in range(7)]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle('Φ Sweep: GNN vs Formula (All Joints)', fontsize=16, fontweight='bold')
    
    for idx in range(7):
        ax = axes[idx // 2, idx % 2]
        ax.plot(phis, formula_all[:, idx], '--', linewidth=2, color='#607D8B', label='Formula', alpha=0.8)
        ax.plot(phis, gnn_all[:, idx], linewidth=2, color='#E91E63', label='GNN', alpha=0.8)
        ax.set_ylabel(f'{joint_names[idx]} (rad)', fontsize=10)
        ax.set_title(joint_names[idx], fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.2)
        ax.axvspan(0, 1, alpha=0.05, color='blue')   # 8_shape
        ax.axvspan(1, 2, alpha=0.05, color='green')   # O_shape
        ax.axvspan(2, 3, alpha=0.05, color='orange')  # door/1
        if idx >= 5:
            ax.set_xlabel('Φ', fontsize=11)
    
    # 最后一个面板: MAE vs Φ
    ax = axes[3, 1]
    mae_per_phi = np.mean(np.abs(gnn_all - formula_all), axis=1)
    ax.fill_between(phis, mae_per_phi, alpha=0.3, color='#E91E63')
    ax.plot(phis, mae_per_phi, linewidth=2, color='#E91E63')
    ax.set_xlabel('Φ', fontsize=11)
    ax.set_ylabel('MAE (rad)', fontsize=10)
    ax.set_title('Mean Abs Error vs Φ', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.axvspan(0, 1, alpha=0.05, color='blue')
    ax.axvspan(1, 2, alpha=0.05, color='green')
    ax.axvspan(2, 3, alpha=0.05, color='orange')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train MorphGNN (Continuous Φ)')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV data file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--model', type=str, default='full', choices=['full', 'lite'])
    parser.add_argument('--aux_weight', type=float, default=0.5, help='Auxiliary raw loss weight')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("   MorphGNN Training (Continuous Φ)")
    print("=" * 60)
    print(f"\n[Config]")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Model: {args.model}")
    print(f"  Input Features: 10 (NodeID=7 + Φ=1 + Lock=1 + Base=1)")
    
    # 查找数据文件
    if args.data is None:
        data_dir = os.path.join(project_root, "data", "collected_datasets")
        csv_files = [f for f in os.listdir(data_dir) if f.startswith('gnn_phi_data') and f.endswith('.csv')]
        if len(csv_files) == 0:
            # 兼容旧格式名
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        csv_files.sort(reverse=True)
        data_path = os.path.join(data_dir, csv_files[0])
    else:
        data_path = args.data
    
    print(f"  Data: {os.path.basename(data_path)}")
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(project_root, "experiments", f"phi_run_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"  Experiment Dir: {exp_dir}")
    
    # 加载数据
    print("\n[Data] Loading dataset...")
    full_dataset = MorphJointDataset(data_path)
    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"  Total: {n_total}, Train: {n_train}, Val: {n_val}")
    
    # 创建模型
    print("\n[Model] Creating model...")
    if args.model == 'full':
        model = MorphGNN().to(device)
    else:
        model = MorphGNNLite().to(device)
    
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    # 训练循环
    print("\n[Training] Starting training...")
    train_losses = []
    val_losses = []
    lr_history = []
    mae_history = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, aux_weight=args.aux_weight)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_history.append(optimizer.param_groups[0]['lr'])
        mae_history.append(val_mae)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(exp_dir, 'best_model.pt'))
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"MAE: {val_mae:.4f} | LR: {lr_now:.1e} | Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, os.path.join(exp_dir, 'final_model.pt'))
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'lr_history': lr_history,
        'mae_history': mae_history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
    }
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 保存配置
    config = {
        'data_path': data_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'model_type': args.model,
        'input_dim': 10,
        'phase_type': 'continuous_phi',
        'n_params': n_params,
        'n_train': n_train,
        'n_val': n_val,
        'device': str(device),
    }
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # =========================================================
    # 生成可视化图表
    # =========================================================
    print("\n[Visualization] Generating plots...")
    
    # 1. 训练仪表盘 (2×2: Loss/LR/MAE/Overfit)
    plot_training_dashboard(train_losses, val_losses, lr_history, mae_history,
                            best_epoch, os.path.join(exp_dir, 'training_dashboard.png'))
    print("  ✓ training_dashboard.png")
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(exp_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. 预测对比图 + 误差分布
    plot_prediction_comparison(model, val_loader, device,
                               os.path.join(exp_dir, 'prediction_analysis.png'))
    print("  ✓ prediction_analysis.png")
    
    # 3. Φ 扫描对比 (GNN vs Formula)
    plot_phi_sweep(model, device, os.path.join(exp_dir, 'phi_sweep.png'))
    print("  ✓ phi_sweep.png")
    
    # 最终评估
    final_loss, final_mae = evaluate(model, val_loader, criterion, device)
    
    # 保存结果
    results = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_val_loss': final_loss,
        'final_val_mae': final_mae,
        'final_lr': lr_history[-1],
        'training_time_seconds': total_time,
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("   Training Complete!")
    print("=" * 60)
    print(f"\n[Results]")
    print(f"  Best Epoch: {best_epoch + 1}")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Final Val MAE: {final_mae:.4f} rad")
    print(f"  Training Time: {total_time:.1f}s")
    print(f"\n[Outputs]")
    print(f"  {exp_dir}/")
    print(f"    ├── best_model.pt")
    print(f"    ├── training_dashboard.png  (Loss/LR/MAE/Overfit)")
    print(f"    ├── prediction_analysis.png (Scatter + Error Histogram)")
    print(f"    ├── phi_sweep.png           (7-Joint GNN vs Formula)")
    print(f"    └── results.json")


if __name__ == "__main__":
    main()

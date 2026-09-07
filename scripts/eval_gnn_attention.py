"""
GNN 注意力热力图分析 (GNN Attention Heatmap)

提取并可视化 GATv2 模型在不同形态（Phi 值）下的注意力权重分布，
展示信息传递是如何随着拓扑形态的变化而自适应改变的。
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, os.path.join(project_root, 'src'))

from models.model import MorphGNN
from utils.graph_converter import RobotGraphConverter

def extract_attention(model, x, edge_index):
    """提取三层 GATv2 的注意力权重"""
    # Layer 1
    h, alpha1 = model.conv1(x, edge_index, return_attention_weights=True)
    h = model.bn1(h)
    h = F.elu(h)
    
    # Layer 2
    h, alpha2 = model.conv2(h, edge_index, return_attention_weights=True)
    h = model.bn2(h)
    h = F.elu(h)
    
    # Layer 3
    h, alpha3 = model.conv3(h, edge_index, return_attention_weights=True)
    
    return [alpha1, alpha2, alpha3]

def get_attention_matrix(alpha_tuple, num_nodes=7):
    """将 (edge_index, attention_weights) 转换为 N x N 邻接矩阵格式的热力图"""
    edges, weights = alpha_tuple
    # weights 维度: [E, heads], 先对 heads 取平均
    if weights.dim() > 1:
        weights = weights.mean(dim=1)
        
    edges = edges.cpu().numpy()
    weights = weights.cpu().detach().numpy()
    
    mat = np.zeros((num_nodes, num_nodes))
    for i in range(edges.shape[1]):
        src = edges[0, i]
        dst = edges[1, i]
        mat[dst, src] = weights[i] # 注意：GAT中，信息是从 src 到 dst，计算 dst 上的注意力
        
    return mat

def plot_heatmap(ax, mat, title, cmap, xticklabels, yticklabels, cbar=False):
    im = ax.imshow(mat, cmap=cmap)
    
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "black" if mat[i, j] < np.max(mat)/2 else "white"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
            
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)
    
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    parser = argparse.ArgumentParser(description='GNN Attention Heatmap')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 加载模型
    if args.model_dir is None:
        exp_base = os.path.join(project_root, "experiments")
        runs = [d for d in os.listdir(exp_base) if d.startswith('phi_run_')]
        if not runs:
            print("❌ No trained model found.")
            return
        runs.sort(reverse=True)
        model_path = os.path.join(exp_base, runs[0], 'best_model.pt')
    else:
        model_path = os.path.join(args.model_dir, 'best_model.pt')
        
    model = MorphGNN().to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {model_path}")
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return
        
    model.eval()
    converter = RobotGraphConverter()
    edge_index = converter.edge_index_7.to(device)
    
    # 选取 3 种典型形态
    phis = [0.5, 1.5, 2.5]
    titles = ['Phi=0.5 (8-shape)', 'Phi=1.5 (O-shape)', 'Phi=2.5 (Door/1-shape)']
    
    # 节点名映射
    node_names = list(converter.MORPH_GRAPH_NODE_NAMES)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(project_root, 'experiments', f'gnn_attention_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    
    # 绘制 Layer 1 的注意力热力图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('GATv2 Attention Weights (Layer 1) across Morphing Phases', fontsize=16, fontweight='bold')
    
    for idx, phi in enumerate(phis):
        x = torch.tensor(converter.get_feature_matrix(phi), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            alphas = extract_attention(model, x, edge_index)
            
        # 我们用 Layer 1 作为分析代表
        mat = get_attention_matrix(alphas[0], num_nodes=7)
        
        plot_heatmap(axes[idx], mat, titles[idx], 'YlGnBu', node_names, node_names, cbar=(idx==2))
        
        axes[idx].set_xlabel('Source Node')
        if idx == 0:
            axes[idx].set_ylabel('Target Node')
            
    plt.tight_layout()
    img_path = os.path.join(out_dir, 'attention_heatmap_layer1.png')
    plt.savefig(img_path, dpi=200)
    plt.close()
    
    # 也可以画一张把三层都画出来的图 (以 Phi=1.5 为例)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Attention Evolution across GNN Layers (Phi=1.5)', fontsize=16, fontweight='bold')
    x = torch.tensor(converter.get_feature_matrix(1.5), dtype=torch.float32).to(device)
    with torch.no_grad():
        alphas = extract_attention(model, x, edge_index)
        
    for layer in range(3):
        mat = get_attention_matrix(alphas[layer], num_nodes=7)
        plot_heatmap(axes[layer], mat, f'Layer {layer+1}', 'Reds', node_names, node_names, cbar=(layer==2))
        
        axes[layer].set_xlabel('Source Node')
        if layer == 0:
            axes[layer].set_ylabel('Target Node')
            
    plt.tight_layout()
    img_path_layer = os.path.join(out_dir, 'attention_evolution_phi1.5.png')
    plt.savefig(img_path_layer, dpi=200)
    plt.close()
    
    print(f"✅ Attention heatmaps saved to: {out_dir}/")

if __name__ == '__main__':
    main()

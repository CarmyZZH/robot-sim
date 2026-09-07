"""
整合三个阶段的完整课程训练学习曲线
运行：python scripts/plot_curriculum_curve.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
exp_base = os.path.join(project_root, 'experiments')

# Phase 1 数据在昨晚第一次完整跑完的目录，Phase 2/3 在今天的目录
PHASES = [
    (
        'Phase 1: Imitation (w=1.0)',
        os.path.join(exp_base, 'drl_curriculum_20260506_221646', 'phase1_imitation', 'monitor.csv'),
        '#4CAF50',
    ),
    (
        'Phase 2: Mixed (w=0.3)',
        os.path.join(exp_base, 'drl_curriculum_20260507_173920', 'phase2_mixed', 'monitor.csv'),
        '#2196F3',
    ),
    (
        'Phase 3: Task Only (w=0.0)',
        os.path.join(exp_base, 'drl_curriculum_20260507_173920', 'phase3_task_only', 'monitor.csv'),
        '#FF5722',
    ),
]
exp_dir = os.path.join(exp_base, 'drl_curriculum_20260507_173920')

# ── 读取数据 ──
dfs = []
for label, path, color in PHASES:
    if not os.path.exists(path):
        print(f"  跳过（文件不存在）: {path}")
        continue
    df = pd.read_csv(path, comment='#')
    df.columns = df.columns.str.strip()
    if len(df) == 0:
        print(f"  跳过（空文件）: {path}")
        continue
    dfs.append((label, df, color))
    r = df['r'].values
    print(f"{label}: {len(df)} episodes, mean={r.mean():.2f}, max={r.max():.2f}")

# ── 画图：2×2 布局 ──
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('PPO Curriculum Training — Full Learning Curve\n(Variable-Topology Robot Brain Layer)',
             fontsize=13, fontweight='bold')

ax_raw    = axes[0, 0]   # 上左：原始+平滑折线
ax_box    = axes[0, 1]   # 上右：箱线图
ax_smooth = axes[1, 0]   # 下左：仅平滑曲线（适合论文）
ax_table  = axes[1, 1]   # 下右：统计表

# ── 上左 & 下左 共用绘图逻辑 ──
for ax, show_raw in [(ax_raw, True), (ax_smooth, False)]:
    offset = 0
    boundaries = []
    for label, df, color in dfs:
        r = df['r'].values
        eps = np.arange(offset, offset + len(r))
        window = max(5, len(r) // 15)
        smoothed = pd.Series(r).rolling(window, min_periods=1).mean().values

        if show_raw:
            ax.plot(eps, r, color=color, alpha=0.15, linewidth=0.6)
        ax.plot(eps, smoothed, color=color, linewidth=2.2, label=label)
        boundaries.append(offset)
        offset += len(r)

    for i, b in enumerate(boundaries[1:], 1):
        ax.axvline(x=b, color='#888', linestyle='--', linewidth=1.0, alpha=0.6)
        ylo = ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -50
        ax.text(b + 3, ylo * 0.85, f'P{i+1}', fontsize=8, color='#666')

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Episode Reward', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.25)

ax_raw.set_title('Episode Reward — Raw + Smoothed', fontsize=10)
ax_smooth.set_title('Smoothed Reward Curve (paper-ready)', fontsize=10)

# ── 上右：箱线图 ──
bp = ax_box.boxplot(
    [df['r'].values for _, df, _ in dfs],
    tick_labels=[lbl.split(':')[0] for lbl, _, _ in dfs],
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2.5),
)
for patch, (_, _, color) in zip(bp['boxes'], dfs):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax_box.axhline(y=0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
ax_box.set_ylabel('Episode Reward', fontsize=10)
ax_box.set_title('Reward Distribution per Phase', fontsize=10)
ax_box.grid(True, alpha=0.25, axis='y')

# ── 下右：统计摘要表 ──
ax_table.axis('off')
header = ['Phase', 'Episodes', 'Mean', 'Max', 'Min', 'Std']
rows = []
row_colors = [['#C8E6C9']*6, ['#BBDEFB']*6, ['#FFCCBC']*6]
for label, df, _ in dfs:
    r = df['r'].values
    rows.append([label.split(':')[0], str(len(r)),
                 f'{r.mean():.1f}', f'{r.max():.1f}',
                 f'{r.min():.1f}', f'{r.std():.1f}'])

tbl = ax_table.table(cellText=rows, colLabels=header,
                     cellColours=row_colors[:len(rows)],
                     loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.15, 2.0)
for j in range(len(header)):
    tbl[(0, j)].set_facecolor('#37474F')
    tbl[(0, j)].get_text().set_color('white')
    tbl[(0, j)].get_text().set_fontweight('bold')
ax_table.set_title('Training Statistics Summary', fontsize=10, pad=14)

plt.tight_layout()
out_path = os.path.join(exp_dir, 'learning_curve_full.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'\n[OK] 图片已保存: {out_path}')

print('\n=== 统计摘要 ===')
for label, df, _ in dfs:
    r = df['r'].values
    print(f"  {label:<38s}  N={len(r):4d}  "
          f"mean={r.mean():7.2f}  max={r.max():6.2f}  std={r.std():.2f}")

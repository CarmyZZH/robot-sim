"""
MPC 轨迹跟踪定量评估 Benchmark

对比三种控制方案:
  1. KinematicMPC (小脑):   预测时域 N=10, 基于优化的最优控制
  2. Pure-Pursuit PID:     经典比例-积分反馈, 无预测
  3. Open-Loop Feedforward: 直接从参考轨迹微分得到速度, 无反馈

测试四种参考轨迹:
  A. 直线 (Straight)
  B. S弯  (S-Curve)
  C. 圆弧 (Circle)
  D. 急转弯 (Sharp Turn)

输出:
  - 多面板轨迹跟踪对比图 (4x3)
  - 误差时间曲线图
  - 定量指标汇总表 (LaTeX-ready)
  - 总结大图 (Bar Chart)

用法:
  python scripts/eval_mpc_benchmark.py
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# 路径
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, os.path.join(project_root, 'src'))

from controllers.mpc_controller import KinematicMPC

# ================================================================
# 全局样式
# ================================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# 控制器对应的颜色与样式
CONTROLLER_STYLES = {
    'MPC':       {'color': '#E91E63', 'lw': 2.2, 'ls': '-',  'label': 'MPC (Ours)'},
    'PID':       {'color': '#2196F3', 'lw': 1.8, 'ls': '-',  'label': 'Pure-Pursuit PID'},
    'OpenLoop':  {'color': '#FF9800', 'lw': 1.5, 'ls': '--', 'label': 'Open-Loop FF'},
}

# ================================================================
# 参考轨迹生成器
# ================================================================

def gen_straight(sim_time, dt, speed=0.3):
    """直线轨迹"""
    steps = int(sim_time / dt)
    t = np.linspace(0, sim_time, steps)
    x = speed * t
    y = np.zeros_like(t)
    yaw = np.zeros_like(t)
    return np.vstack((x, y, yaw)).T

def gen_s_curve(sim_time, dt, speed=0.25, amplitude=0.5, freq=0.15):
    """S弯轨迹"""
    steps = int(sim_time / dt)
    t = np.linspace(0, sim_time, steps)
    x = speed * t
    y = amplitude * np.sin(2 * np.pi * freq * t)
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    yaw = np.arctan2(dy, dx)
    return np.vstack((x, y, yaw)).T

def gen_circle(sim_time, dt, radius=1.0, omega=0.3):
    """圆弧轨迹"""
    steps = int(sim_time / dt)
    t = np.linspace(0, sim_time, steps)
    theta = omega * t
    x = radius * np.sin(theta)
    y = radius * (1 - np.cos(theta))
    yaw = theta  # 切线方向
    return np.vstack((x, y, yaw)).T

def gen_sharp_turn(sim_time, dt, speed=0.25):
    """急转弯轨迹 (L形: 先直走再右转90°)"""
    steps = int(sim_time / dt)
    t = np.linspace(0, sim_time, steps)
    
    t_turn = sim_time * 0.4   # 40% 时间点开始转弯
    t_blend = sim_time * 0.15 # 过渡区宽度
    
    x = np.zeros(steps)
    y = np.zeros(steps)
    yaw = np.zeros(steps)
    
    for i in range(steps):
        ti = t[i]
        if ti < t_turn - t_blend / 2:
            # 直线段: 沿 x 方向
            x[i] = speed * ti
            y[i] = 0.0
            yaw[i] = 0.0
        elif ti < t_turn + t_blend / 2:
            # 过渡段: 平滑转弯
            s = (ti - (t_turn - t_blend / 2)) / t_blend  # s ∈ [0, 1]
            s = 3 * s**2 - 2 * s**3  # smoothstep
            angle = s * (-np.pi / 2)
            x[i] = speed * (t_turn - t_blend / 2) + speed * t_blend * (s - np.sin(s * np.pi / 2) / (np.pi / 2)) * 0.6
            y[i] = -speed * t_blend * (1 - np.cos(s * np.pi / 2)) / (np.pi / 2) * 0.6
            yaw[i] = angle
        else:
            # 转弯后: 沿 -y 方向
            dt_after = ti - (t_turn + t_blend / 2)
            x_at_turn = x[max(0, i-1)]
            y_at_turn = y[max(0, i-1)] if i > 0 else 0
            x[i] = x_at_turn
            y[i] = y_at_turn - speed * dt
            yaw[i] = -np.pi / 2
    
    return np.vstack((x, y, yaw)).T


TRAJECTORIES = {
    'Straight':   {'gen': gen_straight,   'time': 12.0, 'init_offset': [0.0, 0.3, 0.1]},
    'S-Curve':    {'gen': gen_s_curve,    'time': 15.0, 'init_offset': [0.0, 0.5, 0.0]},
    'Circle':     {'gen': gen_circle,     'time': 18.0, 'init_offset': [0.3, 0.0, 0.0]},
    'Sharp-Turn': {'gen': gen_sharp_turn, 'time': 12.0, 'init_offset': [0.0, 0.2, 0.0]},
}

# ================================================================
# 运动学模拟器
# ================================================================

def simulate_unicycle(state, cmd, dt, noise_std=0.0):
    """全向运动学前进积分 (带可选噪声)"""
    x, y, theta = state
    vx, vy, w = cmd
    
    # 加入执行噪声 (模拟打滑等干扰)
    if noise_std > 0:
        vx += np.random.normal(0, noise_std)
        vy += np.random.normal(0, noise_std)
        w += np.random.normal(0, noise_std * 0.5)
    
    x_new = x + (vx * np.cos(theta) - vy * np.sin(theta)) * dt
    y_new = y + (vx * np.sin(theta) + vy * np.cos(theta)) * dt
    theta_new = theta + w * dt
    
    return np.array([x_new, y_new, theta_new])

# ================================================================
# 控制器: Pure-Pursuit PID
# ================================================================

class PurePursuitPID:
    """经典 Pure-Pursuit + PID 速度控制 (无预测, 仅基于当前误差的反馈)"""
    
    def __init__(self, dt=0.1, kp_pos=2.0, kp_yaw=3.0, max_v=0.4, max_w=1.5):
        self.dt = dt
        self.kp_pos = kp_pos
        self.kp_yaw = kp_yaw
        self.max_v = max_v
        self.max_w = max_w
        
    def solve(self, current_state, ref_traj, last_cmd=None):
        """仅使用参考轨迹的第一个点做反馈"""
        x, y, theta = current_state
        ref = ref_traj[0]
        rx, ry, r_yaw = ref
        
        # 位置误差 (世界系)
        dx_w = rx - x
        dy_w = ry - y
        
        # 转到机体系
        dx_b = dx_w * np.cos(theta) + dy_w * np.sin(theta)
        dy_b = -dx_w * np.sin(theta) + dy_w * np.cos(theta)
        
        # 比例控制
        vx = np.clip(self.kp_pos * dx_b, -self.max_v, self.max_v)
        vy = np.clip(self.kp_pos * dy_b, -self.max_v, self.max_v)
        
        # 航向误差
        yaw_err = (r_yaw - theta + np.pi) % (2 * np.pi) - np.pi
        w = np.clip(self.kp_yaw * yaw_err, -self.max_w, self.max_w)
        
        return np.array([vx, vy, w])


class OpenLoopFF:
    """开环前馈: 直接从参考轨迹求微分得到速度, 完全不看当前状态"""
    
    def __init__(self, dt=0.1, max_v=0.4, max_w=1.5):
        self.dt = dt
        self.max_v = max_v
        self.max_w = max_w
        
    def solve(self, current_state, ref_traj, last_cmd=None):
        """用参考轨迹的前两点做数值微分"""
        if len(ref_traj) < 2:
            return np.zeros(3)
        
        dx = ref_traj[1, 0] - ref_traj[0, 0]
        dy = ref_traj[1, 1] - ref_traj[0, 1]
        d_yaw = ref_traj[1, 2] - ref_traj[0, 2]
        d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi
        
        # 转到机体系 (用参考的 yaw, 不用当前实际 yaw)
        ref_theta = ref_traj[0, 2]
        vx_b = (dx * np.cos(ref_theta) + dy * np.sin(ref_theta)) / self.dt
        vy_b = (-dx * np.sin(ref_theta) + dy * np.cos(ref_theta)) / self.dt
        w = d_yaw / self.dt
        
        vx_b = np.clip(vx_b, -self.max_v, self.max_v)
        vy_b = np.clip(vy_b, -self.max_v, self.max_v)
        w = np.clip(w, -self.max_w, self.max_w)
        
        return np.array([vx_b, vy_b, w])


# ================================================================
# 评估指标计算
# ================================================================

def compute_metrics(ref_traj, actual_traj, dt):
    """计算定量评估指标"""
    N = min(len(ref_traj), len(actual_traj))
    ref = ref_traj[:N]
    act = actual_traj[:N]
    
    # 1. Cross-Track Error (横向偏差, 即位置欧氏距离)
    pos_errors = np.sqrt((ref[:, 0] - act[:, 0])**2 + (ref[:, 1] - act[:, 1])**2)
    
    # 2. Heading Error (航向偏差)
    heading_errors = np.abs((ref[:, 2] - act[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    
    # 3. 稳态误差 (最后 20% 的平均误差)
    steady_start = int(N * 0.8)
    
    # 4. 控制平滑度 (加速度 jerk, 用轨迹的二阶差分近似)
    if N > 2:
        d2x = np.diff(act[:, 0], n=2) / dt**2
        d2y = np.diff(act[:, 1], n=2) / dt**2
        jerk = np.sqrt(d2x**2 + d2y**2)
        avg_jerk = np.mean(jerk)
    else:
        avg_jerk = 0.0
    
    # 5. 收敛时间 (第一次误差降到 0.1m 以下的时刻)
    converge_idx = np.argmax(pos_errors < 0.1) if np.any(pos_errors < 0.1) else N
    converge_time = converge_idx * dt
    
    metrics = {
        'avg_cross_track':    np.mean(pos_errors),
        'max_cross_track':    np.max(pos_errors),
        'rms_cross_track':    np.sqrt(np.mean(pos_errors**2)),
        'avg_heading_err':    np.degrees(np.mean(heading_errors)),
        'max_heading_err':    np.degrees(np.max(heading_errors)),
        'steady_state_err':   np.mean(pos_errors[steady_start:]),
        'avg_jerk':           avg_jerk,
        'converge_time':      converge_time,
        'pos_errors':         pos_errors,           # 时间序列 (用于画图)
        'heading_errors':     np.degrees(heading_errors),  # 时间序列
    }
    return metrics


# ================================================================
# 运行单次仿真
# ================================================================

def run_simulation(controller, ref_traj, init_state, dt, horizon=10, noise_std=0.005):
    """运行一次完整的轨迹跟踪仿真"""
    current_state = init_state.copy()
    actual_traj = [current_state.copy()]
    last_cmd = np.zeros(3)
    cmds = []
    solve_times = []
    
    total_steps = len(ref_traj) - horizon
    
    for k in range(total_steps):
        local_ref = ref_traj[k: k + horizon]
        if len(local_ref) < 2:
            break
        
        t0 = time.perf_counter()
        cmd = controller.solve(current_state, local_ref, last_cmd)
        solve_time = (time.perf_counter() - t0) * 1000  # ms
        solve_times.append(solve_time)
        
        last_cmd = cmd
        cmds.append(cmd.copy())
        current_state = simulate_unicycle(current_state, cmd, dt, noise_std=noise_std)
        actual_traj.append(current_state.copy())
    
    return {
        'actual_traj': np.array(actual_traj),
        'cmds': np.array(cmds) if cmds else np.zeros((0, 3)),
        'avg_solve_time': np.mean(solve_times) if solve_times else 0.0,
    }


# ================================================================
# 主评估流程
# ================================================================

def main():
    print("=" * 70)
    print("   MPC Trajectory Tracking Benchmark Evaluation")
    print("=" * 70)
    
    dt = 0.1
    horizon = 10
    noise_std = 0.005   # 轻微执行噪声
    
    # 创建输出目录
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, 'experiments', f'mpc_benchmark_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 实例化三种控制器
    controllers = {
        'MPC':      KinematicMPC(dt=dt, N=horizon),
        'PID':      PurePursuitPID(dt=dt),
        'OpenLoop': OpenLoopFF(dt=dt),
    }
    
    # 存储全部结果
    all_results = {}  # {traj_name: {ctrl_name: {metrics, sim_result}}}
    
    # ============================================================
    # 逐轨迹 × 逐控制器 运行仿真
    # ============================================================
    for traj_name, traj_cfg in TRAJECTORIES.items():
        ref_traj = traj_cfg['gen'](traj_cfg['time'], dt)
        offset = np.array(traj_cfg['init_offset'])
        init_state = ref_traj[0].copy() + offset
        
        print(f"\n{'─' * 50}")
        print(f"  Trajectory: {traj_name}")
        print(f"  Duration: {traj_cfg['time']}s, Points: {len(ref_traj)}")
        print(f"  Initial offset: dx={offset[0]:.2f}, dy={offset[1]:.2f}, dyaw={offset[2]:.2f}")
        
        all_results[traj_name] = {}
        
        for ctrl_name, ctrl in controllers.items():
            # 重置 MPC 热启动
            if hasattr(ctrl, 'u_prev'):
                ctrl.u_prev = np.zeros(horizon * 3)
            
            sim = run_simulation(ctrl, ref_traj, init_state, dt, horizon, noise_std)
            metrics = compute_metrics(ref_traj, sim['actual_traj'], dt)
            metrics['avg_solve_time'] = sim['avg_solve_time']
            
            all_results[traj_name][ctrl_name] = {
                'sim': sim,
                'metrics': metrics,
                'ref_traj': ref_traj,
            }
            
            print(f"    [{ctrl_name:>8s}] RMS_CTE={metrics['rms_cross_track']:.4f}m  "
                  f"AvgHeading={metrics['avg_heading_err']:.2f}°  "
                  f"SteadyState={metrics['steady_state_err']:.4f}m  "
                  f"SolveTime={metrics['avg_solve_time']:.2f}ms")
    
    # ============================================================
    # 图1: 四轨迹 × 三控制器 轨迹跟踪对比 (4×1 子图)
    # ============================================================
    print(f"\n{'=' * 50}")
    print("  Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Trajectory Tracking Comparison: MPC vs PID vs Open-Loop',
                 fontsize=16, fontweight='bold', y=0.98)
    
    for idx, (traj_name, results) in enumerate(all_results.items()):
        ax = axes[idx // 2, idx % 2]
        
        ref_traj = results['MPC']['ref_traj']
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'k--', linewidth=1.5, alpha=0.6, label='Reference', zorder=1)
        
        for ctrl_name in ['OpenLoop', 'PID', 'MPC']:
            style = CONTROLLER_STYLES[ctrl_name]
            act = results[ctrl_name]['sim']['actual_traj']
            ax.plot(act[:, 0], act[:, 1], 
                    color=style['color'], linewidth=style['lw'], linestyle=style['ls'],
                    label=style['label'], zorder=2)
        
        # 起点
        ax.scatter(ref_traj[0, 0], ref_traj[0, 1], c='green', s=80, zorder=5, marker='o', edgecolors='black', linewidths=0.5)
        ax.scatter(ref_traj[-1, 0], ref_traj[-1, 1], c='red', s=80, zorder=5, marker='s', edgecolors='black', linewidths=0.5)
        
        ax.set_title(f'{traj_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='best', fontsize=8, framealpha=0.8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path1 = os.path.join(output_dir, 'fig1_trajectory_comparison.png')
    plt.savefig(path1)
    plt.close()
    print(f"  ✓ {os.path.basename(path1)}")
    
    # ============================================================
    # 图2: 误差时间曲线 (4×2: Cross-Track + Heading)
    # ============================================================
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('Tracking Error Over Time', fontsize=16, fontweight='bold', y=0.99)
    
    for idx, (traj_name, results) in enumerate(all_results.items()):
        ax_cte = axes[idx, 0]
        ax_head = axes[idx, 1]
        
        for ctrl_name in ['OpenLoop', 'PID', 'MPC']:
            style = CONTROLLER_STYLES[ctrl_name]
            m = results[ctrl_name]['metrics']
            t_axis = np.arange(len(m['pos_errors'])) * dt
            
            ax_cte.plot(t_axis, m['pos_errors'],
                       color=style['color'], linewidth=style['lw'], linestyle=style['ls'],
                       label=style['label'])
            ax_head.plot(t_axis[:len(m['heading_errors'])], m['heading_errors'],
                        color=style['color'], linewidth=style['lw'], linestyle=style['ls'],
                        label=style['label'])
        
        ax_cte.set_ylabel('Cross-Track Error (m)')
        ax_cte.set_title(f'{traj_name} — Position Error', fontsize=11, fontweight='bold')
        ax_cte.grid(True, alpha=0.2)
        ax_cte.legend(fontsize=8)
        ax_cte.set_ylim(bottom=0)
        
        ax_head.set_ylabel('Heading Error (°)')
        ax_head.set_title(f'{traj_name} — Heading Error', fontsize=11, fontweight='bold')
        ax_head.grid(True, alpha=0.2)
        ax_head.legend(fontsize=8)
        ax_head.set_ylim(bottom=0)
        
        if idx == 3:
            ax_cte.set_xlabel('Time (s)')
            ax_head.set_xlabel('Time (s)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path2 = os.path.join(output_dir, 'fig2_error_curves.png')
    plt.savefig(path2)
    plt.close()
    print(f"  ✓ {os.path.basename(path2)}")
    
    # ============================================================
    # 图3: 汇总柱状图 (Bar Chart)
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Quantitative Performance Summary', fontsize=14, fontweight='bold')
    
    traj_names = list(all_results.keys())
    ctrl_names = ['MPC', 'PID', 'OpenLoop']
    colors = [CONTROLLER_STYLES[c]['color'] for c in ctrl_names]
    x = np.arange(len(traj_names))
    width = 0.25
    
    # (a) RMS CTE
    ax = axes[0]
    for i, cn in enumerate(ctrl_names):
        vals = [all_results[tn][cn]['metrics']['rms_cross_track'] for tn in traj_names]
        bars = ax.bar(x + i * width, vals, width, color=colors[i], 
                      label=CONTROLLER_STYLES[cn]['label'], edgecolor='white', linewidth=0.5)
        # 数值标注
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('RMS Cross-Track Error (m)')
    ax.set_title('(a) Position Accuracy')
    ax.set_xticks(x + width)
    ax.set_xticklabels(traj_names, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    
    # (b) Avg Heading Error
    ax = axes[1]
    for i, cn in enumerate(ctrl_names):
        vals = [all_results[tn][cn]['metrics']['avg_heading_err'] for tn in traj_names]
        bars = ax.bar(x + i * width, vals, width, color=colors[i],
                      label=CONTROLLER_STYLES[cn]['label'], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('Avg Heading Error (°)')
    ax.set_title('(b) Heading Accuracy')
    ax.set_xticks(x + width)
    ax.set_xticklabels(traj_names, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    
    # (c) Avg Solve Time
    ax = axes[2]
    for i, cn in enumerate(ctrl_names):
        vals = [all_results[tn][cn]['metrics']['avg_solve_time'] for tn in traj_names]
        bars = ax.bar(x + i * width, vals, width, color=colors[i],
                      label=CONTROLLER_STYLES[cn]['label'], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('Avg Solve Time (ms)')
    ax.set_title('(c) Computational Cost')
    ax.set_xticks(x + width)
    ax.set_xticklabels(traj_names, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    path3 = os.path.join(output_dir, 'fig3_summary_bars.png')
    plt.savefig(path3)
    plt.close()
    print(f"  ✓ {os.path.basename(path3)}")
    
    # ============================================================
    # 打印数据表 (可直接粘贴到论文)
    # ============================================================
    print(f"\n{'=' * 90}")
    print("  QUANTITATIVE RESULTS TABLE")
    print(f"{'=' * 90}")
    
    header = f"{'Trajectory':<12s} │ {'Controller':<12s} │ {'RMS_CTE(m)':>10s} │ {'MaxCTE(m)':>9s} │ {'AvgHead(°)':>10s} │ {'SS_Err(m)':>9s} │ {'Jerk':>8s} │ {'T_conv(s)':>9s} │ {'T_solve(ms)':>11s}"
    print(header)
    print("─" * len(header))
    
    for traj_name in traj_names:
        for ctrl_name in ctrl_names:
            m = all_results[traj_name][ctrl_name]['metrics']
            print(f"{traj_name:<12s} │ {ctrl_name:<12s} │ "
                  f"{m['rms_cross_track']:>10.4f} │ "
                  f"{m['max_cross_track']:>9.4f} │ "
                  f"{m['avg_heading_err']:>10.2f} │ "
                  f"{m['steady_state_err']:>9.4f} │ "
                  f"{m['avg_jerk']:>8.2f} │ "
                  f"{m['converge_time']:>9.2f} │ "
                  f"{m['avg_solve_time']:>11.2f}")
        print("─" * len(header))
    
    # ============================================================
    # 图4: MPC 独立细节图 (带朝向箭头 + 误差条)
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MPC Tracking Detail (with Heading Arrows)', fontsize=14, fontweight='bold', y=0.98)
    
    for idx, (traj_name, results) in enumerate(all_results.items()):
        ax = axes[idx // 2, idx % 2]
        
        ref = results['MPC']['ref_traj']
        act = results['MPC']['sim']['actual_traj']
        
        ax.plot(ref[:, 0], ref[:, 1], 'k--', linewidth=1.5, alpha=0.5, label='Reference')
        ax.plot(act[:, 0], act[:, 1], color='#E91E63', linewidth=2, label='MPC Actual')
        
        # 每隔一段绘制朝向箭头和误差线
        step = max(1, len(act) // 15)
        for i in range(0, min(len(act), len(ref)), step):
            # 误差连线
            ax.plot([ref[i, 0], act[i, 0]], [ref[i, 1], act[i, 1]], 
                    color='gray', linewidth=0.8, alpha=0.4)
            # 机器人朝向
            arrow_len = 0.12
            ax.annotate('', xy=(act[i, 0] + arrow_len * np.cos(act[i, 2]),
                                act[i, 1] + arrow_len * np.sin(act[i, 2])),
                        xytext=(act[i, 0], act[i, 1]),
                        arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.2))
        
        ax.scatter(ref[0, 0], ref[0, 1], c='green', s=80, zorder=5, marker='o', edgecolors='black')
        ax.scatter(ref[-1, 0], ref[-1, 1], c='red', s=80, zorder=5, marker='s', edgecolors='black')
        
        m = results['MPC']['metrics']
        text = f"RMS CTE: {m['rms_cross_track']:.4f} m\nSS Err: {m['steady_state_err']:.4f} m"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{traj_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path4 = os.path.join(output_dir, 'fig4_mpc_detail.png')
    plt.savefig(path4)
    plt.close()
    print(f"  ✓ {os.path.basename(path4)}")
    
    # ============================================================
    # 完成
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  All outputs saved to: {output_dir}/")
    print(f"    ├── fig1_trajectory_comparison.png")
    print(f"    ├── fig2_error_curves.png")
    print(f"    ├── fig3_summary_bars.png")
    print(f"    └── fig4_mpc_detail.png")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    np.random.seed(42)
    main()

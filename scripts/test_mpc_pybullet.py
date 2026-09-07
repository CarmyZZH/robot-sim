import sys
import os
import time
import numpy as np
import pybullet as p
import torch
import matplotlib.pyplot as plt
from collections import deque

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from envs.robot_env import MorphingRobotEnv
from controllers.swerve_kinematics import SwerveDriveController

class ZeroYawSmoother:
    """EMA filter to smooth out physics engine jitter on the zero-yaw readings"""
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.state = {}
        
    def update(self, current_yaws):
        smoothed = {}
        for corner, yaw in current_yaws.items():
            if corner not in self.state:
                self.state[corner] = yaw
            else:
                prev = self.state[corner]
                diff = (yaw - prev + np.pi) % (2 * np.pi) - np.pi
                self.state[corner] = prev + self.alpha * diff
            smoothed[corner] = self.state[corner]
        return smoothed

def compute_angles_from_phi(phi: float) -> list:
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

# =========================================================
# GNN 模型加载逻辑
# =========================================================
def load_gnn_model(device):
    try:
        from models.model import MorphGNN
        from utils.graph_converter import RobotGraphConverter
        exp_base = os.path.join(project_root, "experiments")
        exp_dirs = sorted([d for d in os.listdir(exp_base) if d.startswith('phi_run_')], reverse=True)
        if not exp_dirs: return None, None
        model_path = os.path.join(exp_base, exp_dirs[0], "best_model.pt")
        model = MorphGNN().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        converter = RobotGraphConverter()
        print(f"[GNN] ✓ Loaded model: {exp_dirs[0]}")
        return model, converter
    except Exception as e:
        print("[GNN] ✗ Failed to load:", e)
        return None, None

def gnn_predict(model, converter, phi, edge_index, device):
    features = converter.get_feature_matrix(phi)
    x = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred, _ = model(x, edge_index)
    g = pred.squeeze(-1).cpu().numpy()
    return converter.graph_to_env_angles(g)

def run_gnn_evaluation_demo():
    urdf_path = os.path.join(project_root, "data", "my_robot", "robot1231_2.urdf")
    
    # 1. 环境初始化
    env = MorphingRobotEnv(urdf_path, gui=True, control_steps=1)
    env.reset()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model, gnn_converter = load_gnn_model(device)
    edge_index = gnn_converter.edge_index_7.to(device) if gnn_model else None
    
    swerve = SwerveDriveController(wheel_radius=env.WHEEL_RADIUS)
    zero_yaw_smoother = ZeroYawSmoother(alpha=0.15)
    
    # 2. 图形画板深度美化 (单大面板独占)
    try:
        plt.style.use('ggplot')
    except:
        pass  # 兼容没装 ggplot 主题的环境
        
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.canvas.manager.set_window_title("GNN Structural Inference Analysis")

    ax.set_title("Neural Network Forecast vs. Theoretical Benchmark (Real-time 7-DOF Analysis)", 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Joint Angles (rad)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(-2.2, 2.2)
    ax.grid(color='white', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.set_facecolor('#F0F0F5')  # 高级柔和灰底色

    # 七种明艳且极其有辨识度的 Material Colors
    colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#795548']
    
    lines_gnn = []
    lines_perfect = []
    dt_tick = 0.05
    
    for i in range(7):
        # 实线再度减细调柔和
        l_gnn, = ax.plot([], [], color=colors[i], linestyle='-', linewidth=1.2, alpha=0.95, 
                         label=f"J{i+1} GNN")
        # 虚线作为基准背景，保持极致纤细
        l_perf, = ax.plot([], [], color=colors[i], linestyle='--', linewidth=0.8, alpha=0.5, 
                          label=f"J{i+1} Math")
        lines_gnn.append(l_gnn)
        lines_perfect.append(l_perf)
        
    # 图例水平铺开，放置于图表正下方（分2行，每排7列）
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=7, fontsize=10, 
              frameon=True, facecolor='white', shadow=True, edgecolor='#cccccc')
    
    # 扩展右侧空域，给底部预留高度容纳双排图例
    plt.subplots_adjust(right=0.95, left=0.08, top=0.9, bottom=0.25)
    
    t_data = [] # 记录实线经过的时间长度
    gnn_data = [[] for _ in range(7)] # 记录经过的真实推断数据
    
    # ★ 提前计算好整整 12 秒钟发生的所有未来真值，并一瞬间“铺底绘制”成功！
    ax.set_xlim(-0.2, 12.2)
    t_full = np.arange(0, 12.0 + dt_tick, dt_tick)
    for i in range(7):
        y_perf = []
        for t_curr in t_full:
            if t_curr <= 3.0:
                phi_opt = 1.5 - (1.5 / 3.0) * t_curr
            elif t_curr <= 4.0:
                phi_opt = 0.0
            else:
                phi_opt = 0.0 + (3.0 / 6.0) * (t_curr - 4.0)
            y_perf.append(compute_angles_from_phi(phi_opt)[i])
        # 直接静死锁在屏幕上形成整场预判
        lines_perfect[i].set_data(t_full, y_perf)
        
    fig.canvas.draw()

    # 3. 等待 20s 真实时钟让系统落稳
    print("=========================================")
    print("  [Auto] 图表已就位！")
    print("  等待 20 秒钟自然时间让物理引擎稳固着陆...")
    print("=========================================")
    init_phi = 1.5
    init_blocks = compute_angles_from_phi(init_phi)
    wait_start = time.time()
    
    while time.time() - wait_start < 30.0:
        env.step({'blocks': init_blocks, 'velocity': [0]*4, 'steering': [0]*4})
        # 【关键】在此处不断刷新 matplotlib 事件队列以防系统将其判定为卡死
        fig.canvas.flush_events() 
        time.sleep(1.0/240.0)

    # 4. 执行 12 秒钟全自动评估循环
    print("=========================================")
    print("  [Auto] 15秒已过，开启 12s 全动形变直线猛冲测试...")
    print("=========================================")
    
    step_count = 0
    total_steps = int(12.0 / dt_tick) # 240 步
    p.removeAllUserDebugItems()

    while step_count < total_steps:
        current_t = step_count * dt_tick
        
        # A. 时间序列 - 形态变换 (1.5 -> 0, 等待, 0 -> 3)
        if current_t <= 3.0:
            phi = 1.5 - (1.5 / 3.0) * current_t
        elif current_t <= 4.0:
            phi = 0.0
        else:
            phi = 0.0 + (3.0 / 6.0) * (current_t - 4.0)

        # 获取完美解和神经网络推算解
        formula_angles = compute_angles_from_phi(phi)
        if gnn_model:
            gnn_angles = gnn_predict(gnn_model, gnn_converter, phi, edge_index, device)
        else:
            gnn_angles = formula_angles
        
        block_angles = gnn_angles

        # B. 绝对恒定 0.4 m/s (斜对角跑动，替代 MPC 曲线)
        cmd_vx_w = 0.4
        cmd_vy_w = 0.4
        cmd_omega = 0.0
        
        # C. 驱动层转换
        raw_zero_yaws = env.get_steering_zero_yaws_imu()
        zero_yaws_w = zero_yaw_smoother.update(raw_zero_yaws)
        
        for i, s_id in enumerate(env.steering_ids):
            corner = env.STEER_CORNER_ORDER[i]
            swerve.last_angles[corner] = p.getJointState(env.robotId, s_id)[0]
            
        is_stationary = (abs(cmd_vx_w) < 0.01 and abs(cmd_vy_w) < 0.01 and abs(cmd_omega) < 0.03)
        steers, vels = [], []
        if is_stationary:
            for i in range(len(env.steering_ids)):
                steers.append(swerve.last_angles[env.STEER_CORNER_ORDER[i]])
                vels.append(0.0)
        else:
            wheel_positions_world, joint_to_corner = env.get_wheel_positions_for_kinematics()
            cmds = swerve.compute_swerve_kinematics(
                cmd_vx_w, cmd_vy_w, cmd_omega, wheel_positions_world, zero_yaws_w
            )
            for i in range(len(env.steering_ids)):
                corner = joint_to_corner[i]
                data = cmds.get(corner, {'angle': 0, 'speed': 0})
                steers.append(data['angle'])
                vels.append(data['speed'])
            
        action = {'steering': steers, 'velocity': vels, 'blocks': block_angles}
        
        # D. 执行物理步伐
        env.step(action)
        env.step(action)
        
        # E. 更新图画板 (追加数据，每三步重绘以丝滑抗抖)
        t_data.append(current_t)
        for i in range(7):
            gnn_data[i].append(gnn_angles[i])
            
        if step_count % 3 == 0:
            for i in range(7):
                # 仅推进实线向前蔓延覆盖虚线，再也不动坐标轴边缘
                lines_gnn[i].set_data(t_data, gnn_data[i])
            fig.canvas.flush_events()
            
        step_count += 1
        
    print("=========================================")
    print("  [Finished] 12s 组会实验落成！")
    print("  底层图形界面不再刷新并持久化留存，随意截图取用...")
    print("=========================================")
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    run_gnn_evaluation_demo()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import matplotlib.pyplot as plt
from controllers.mpc_controller import KinematicMPC

def generate_reference_trajectory(sim_time, dt):
    """
    生成一个连续平滑的测试参考轨迹 (例如 8字形 / S弯)
    返回: [x, y, yaw] 的序列
    """
    steps = int(sim_time / dt)
    t = np.linspace(0, sim_time, steps)
    
    # 一个简单的正弦曲线轨迹
    x_ref = 1.0 * t
    y_ref = 2.0 * np.sin(0.5 * t)
    
    # 算术求导得到期望的偏航角 (切线方向)
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    yaw_ref = np.arctan2(dy, dx)
    
    return np.vstack((x_ref, y_ref, yaw_ref)).T

def simulate_unicycle(state, cmd, dt):
    """基于 Unicycle 运动学更新机器人真实状态"""
    x, y, theta = state
    vx, vy, w = cmd
    
    # 机体速度 -> 世界速度 -> 积分
    x_new = x + (vx * np.cos(theta) - vy * np.sin(theta)) * dt
    y_new = y + (vx * np.sin(theta) + vy * np.cos(theta)) * dt
    theta_new = theta + w * dt
    
    return np.array([x_new, y_new, theta_new])

def main():
    print("Initializing Light Kinematic MPC...")
    dt = 0.1
    horizon = 10
    mpc = KinematicMPC(dt=dt, N=horizon)
    
    sim_time = 15.0
    ref_traj = generate_reference_trajectory(sim_time, dt)
    
    # 机器人初始位置 (故意给一个很大的初始偏差，看 MPC 能不能纠正回来)
    current_state = np.array([0.0, -3.0, np.pi/2]) 
    
    actual_traj = []
    actual_traj.append(current_state)
    last_cmd = np.zeros(3)
    
    print("Running MPC Tracking Simulation...")
    for k in range(len(ref_traj) - horizon):
        # 截取未来 N 步的参考目标
        local_ref = ref_traj[k : k + horizon]
        
        # MPC 求解最优指令
        cmd = mpc.solve(current_state, local_ref, last_cmd)
        last_cmd = cmd
        
        # 物理模拟：用算出来的指令去移动机器人
        # 我们可以在这里加入一点随机噪声模拟打滑
        current_state = simulate_unicycle(current_state, cmd, dt)
        actual_traj.append(current_state)
        
        # 进度条
        if k % 10 == 0:
            print(f"[{k}/{len(ref_traj) - horizon}] Tracking Error: "
                  f"dx={local_ref[0,0]-current_state[0]:.2f}, "
                  f"dy={local_ref[0,1]-current_state[1]:.2f}")

    actual_traj = np.array(actual_traj)
    
    print("Simulation finished. Plotting results...")
    
    # 可视化对比
    plt.figure(figsize=(10, 6))
    
    # 绘制参考轨迹
    plt.plot(ref_traj[:, 0], ref_traj[:, 1], 'k--', linewidth=2, label="Reference Trajectory (Brain)")
    
    # 绘制真实轨迹
    plt.plot(actual_traj[:, 0], actual_traj[:, 1], 'b-', linewidth=2, label="Actual Path (via MPC Cerebellum)")
    
    # 绘制机器人的朝向箭头以显示姿态
    for i in range(0, len(actual_traj), 15):
        plt.arrow(actual_traj[i, 0], actual_traj[i, 1], 
                  0.3*np.cos(actual_traj[i, 2]), 0.3*np.sin(actual_traj[i, 2]), 
                  head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        plt.arrow(ref_traj[i, 0], ref_traj[i, 1], 
                  0.3*np.cos(ref_traj[i, 2]), 0.3*np.sin(ref_traj[i, 2]), 
                  head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.5)

    # 标出起点
    plt.scatter(actual_traj[0, 0], actual_traj[0, 1], c='red', s=100, label="Start Position")
    
    plt.title("Light Kinematic MPC Tracking Demonstration")
    plt.xlabel("World X (m)")
    plt.ylabel("World Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 自动保存图片，方便无 GUI 环境下查看
    save_path = os.path.join(os.path.dirname(__file__), 'mpc_tracking_result.png')
    plt.savefig(save_path)
    print(f"Result plotted and saved to: {save_path}")
    plt.show()

if __name__ == '__main__':
    main()

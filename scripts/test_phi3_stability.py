import sys
import os
import time
import numpy as np
import pybullet as p

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

from envs.robot_env import MorphingRobotEnv
from controllers.swerve_kinematics import SwerveDriveController
from scripts.test_mpc_vs_pid_topology import ZeroYawSmoother, compute_angles_from_phi

def main():
    print("==================================================")
    print("  Phi=3.0 (利剑模式) 纯净平地稳定性测试 Demo")
    print("==================================================")
    
    urdf_path = os.path.join(project_root, "data", "my_robot", "robot1231_2.urdf")
    # 直接使用底层物理环境，不带任何场景障碍物
    env = MorphingRobotEnv(urdf_path, gui=True, control_steps=12)
    swerve = SwerveDriveController(wheel_radius=env.WHEEL_RADIUS)
    
    # ---------------------------------------------------------
    # 1. 初始平滑形变准备 (1.5 -> 3.0)
    # ---------------------------------------------------------
    env.reset()
    current_phi = 1.5
    target_phi_final = 3.0
    
    print("\n[Prep] 正在进行平滑形变至 Phi=3.0...")
    while abs(current_phi - target_phi_final) > 0.01:
        current_phi += 0.01
        target_blocks = compute_angles_from_phi(current_phi)
        env.step({'blocks': target_blocks, 'velocity': [0]*4, 'steering': [0]*4})
        time.sleep(0.01)
        
    print("[Prep] 形变完成，锁定 Phi=3.0。")
    target_blocks_phi3 = compute_angles_from_phi(3.0)
    
    print("\n--- 开始 0.4m/s 平地稳定性测试 ---")
    print("Step | Raw_Yaw (rad) | Filtered_Yaw (rad) | Jitter_Delta")
    print("-" * 65)
    
    last_filtered_yaw = 0.0
    alpha = 0.1 # 建议的低通滤波系数
    
    target_vx_w = 0.4 # 0.4m/s forward
    target_vy_w = 0.0
    target_omega = 0.0
    
    try:
        for step in range(500):
            # 直接通过底层接口获取位姿
            x, y, raw_yaw = env.get_robot_pose()
            
            # 初始化第一次的滤波值
            if step == 0:
                last_filtered_yaw = raw_yaw
            
            # 1. 角度增量滤波算法 (处理 -pi/pi 突变)
            diff = raw_yaw - last_filtered_yaw
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            filtered_yaw = last_filtered_yaw + alpha * diff
            last_filtered_yaw = (filtered_yaw + np.pi) % (2 * np.pi) - np.pi
            
            # 2. 运动控制 (直线行驶)
            raw_zero_yaws = env.get_steering_zero_yaws_imu()
            cmds_sw = swerve.compute_swerve_kinematics(target_vx_w, target_vy_w, target_omega, 
                                                    env.get_wheel_positions_for_kinematics()[0], raw_zero_yaws)
                                                    
            steers = [cmds_sw[env.STEER_CORNER_ORDER[i]]['angle'] for i in range(4)]
            vels = [cmds_sw[env.STEER_CORNER_ORDER[i]]['speed'] for i in range(4)]
            
            action = {'steering': steers, 'velocity': vels, 'blocks': target_blocks_phi3}
            # MorphingRobotEnv.step 返回的是 (obs, reward, done, info)
            _ = env.step(action)
            
            # 3. 数据记录与可视化
            jitter_delta = abs(raw_yaw - last_filtered_yaw)
            if step % 20 == 0:
                print(f"{step:03d}  |  {raw_yaw:+.6f}  |  {last_filtered_yaw:+.6f}  |  {jitter_delta:.6f}")
            
            # 绘制可视化箭头
            p.removeAllUserDebugItems()
            # 原始车头 (红色)
            p.addUserDebugLine([x, y, 0.2], [x+0.5*np.cos(raw_yaw), y+0.5*np.sin(raw_yaw), 0.2], [1,0,0], 3)
            # 滤波平滑后的车头 (绿色)
            p.addUserDebugLine([x, y, 0.25], [x+0.5*np.cos(last_filtered_yaw), y+0.5*np.sin(last_filtered_yaw), 0.25], [0,1,0], 5)
            
            time.sleep(0.005)
            
    except KeyboardInterrupt:
        pass

    print("\n测试完成。")

if __name__ == "__main__":
    main()

import sys
import os
import time
import numpy as np
import pybullet as p

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root) 

from envs.benchmark_env import BenchmarkEnv
from brains.scripted_brain import ScriptedBrain
from controllers.mpc_controller import KinematicMPC
from controllers.swerve_kinematics import SwerveDriveController
from scripts.test_mpc_vs_pid_topology import ZeroYawSmoother, compute_angles_from_phi
import torch

def load_gnn_model(device):
    try:
        from models.model import MorphGNN
        from utils.graph_converter import RobotGraphConverter
        
        exp_base = os.path.join(project_root, "experiments")
        if not os.path.exists(exp_base): return None, None
        
        exp_dirs = sorted([d for d in os.listdir(exp_base) if d.startswith('phi_run_')], reverse=True)
        if not exp_dirs: return None, None
        
        model_path = os.path.join(exp_base, exp_dirs[0], "best_model.pt")
        if not os.path.exists(model_path): return None, None
        
        model = MorphGNN().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        converter = RobotGraphConverter()
        print(f"[GNN] [OK] 成功加载模型: {os.path.basename(os.path.dirname(model_path))}")
        return model, converter
    except Exception as e:
        print(f"[GNN] [FAIL] 加载失败: {e}")
        return None, None

def gnn_predict(model, converter, phi, edge_index, device):
    features = converter.get_feature_matrix(phi)
    x = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred, _ = model(x, edge_index)
    g = pred.squeeze(-1).cpu().numpy()
    return converter.graph_to_env_angles(g)

def main():
    print("==================================================")
    print(" 启动全链路：三层架构合体评估 (Benchmark Arena) ")
    print("==================================================")
    print(" 请选择运行模式：")
    print("  [1] 机器人闭环仿真模式 (Robot Simulation Mode)")
    print("  [2] ABC场景选择展示 (Scene Viewer)")
    
    try:
        main_menu = int(input("请输入模式编号 (1或2) [默认1]: ").strip())
    except:
        main_menu = 1
        
    urdf_path = os.path.join(project_root, "data", "my_robot", "robot1231_2.urdf")
    
    # ----------------------------------------------------------------------
    # 模式 1: 机器人仿真模式
    # ----------------------------------------------------------------------
    if main_menu == 1:
        print("\n--- 机器人闭环仿真模式 (默认场景: Scene C) ---")
        selected_scene = 'C'
        print(" 请选择测试赛制：")
        print("  [1] 智慧过关阵营 (Smart Morphing Brain)")
        print("  [2] 刚直碰撞阵营 (Ablation Test - 永远保持O型)")
        try:
            mode = int(input("请输入赛制编号 (1或2) [默认1]: ").strip())
        except:
            mode = 1
            
        env = BenchmarkEnv(urdf_path, gui=True, scene_id=selected_scene)

        # ---------------------------------------------------------
        # 【物理级硬锁定补丁】: 弃用大力矩，改用 resetJointState 强行封存角度
        # ---------------------------------------------------------
        def adaptive_pivot_hold(self_env):
            is_morphing = getattr(self_env, 'dynamic_pivot_is_morphing', True)
            is_crossing = getattr(self_env, 'dynamic_pivot_is_crossing', False)
            should_lock = (not is_morphing) or is_crossing
            
            if should_lock:
                if getattr(self_env, 'pivot_anchors', None) is None:
                    self_env.pivot_anchors = [p.getJointState(self_env.robotId, pid)[0] for pid in self_env.pivot_ids]
            else:
                self_env.pivot_anchors = None
                
            for i, p_id in enumerate(self_env.pivot_ids):
                if not should_lock or self_env.pivot_anchors is None:
                    # 变身模式：极低力矩随动
                    curr_pos = p.getJointState(self_env.robotId, p_id)[0]
                    p.setJointMotorControl2(self_env.robotId, p_id, p.POSITION_CONTROL, targetPosition=curr_pos, force=0.5)
                else:
                    # 硬锁定模式：直接重置关节状态，实现上帝层面的位姿封存
                    target = self_env.pivot_anchors[i]
                    p.setJointMotorControl2(self_env.robotId, p_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
                    p.resetJointState(self_env.robotId, p_id, targetValue=target, targetVelocity=0)

        import types
        env.core_env.update_pivot_hold = types.MethodType(adaptive_pivot_hold, env.core_env)
        env.core_env.dynamic_pivot_is_crossing = False
        env.core_env.pivot_anchors = None
        # ---------------------------------------------------------

        brain = ScriptedBrain(dt=0.05, mpc_horizon=10)
        mpc = KinematicMPC(dt=0.05, N=10)
        swerve = SwerveDriveController(wheel_radius=env.core_env.WHEEL_RADIUS)
        smoother = ZeroYawSmoother(alpha=0.9) # 极高动态响应，消除转体时的舵轮滞后
        
        global_traj = env.task.global_traj 
        obs = env.reset()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gnn_model, gnn_converter = load_gnn_model(device)
        edge_index = gnn_converter.edge_index_7.to(device) if gnn_model is not None else None
        
        if gnn_model is not None:
            init_blocks = gnn_predict(gnn_model, gnn_converter, 1.5, edge_index, device)
        else:
            init_blocks = compute_angles_from_phi(1.5)
            
        for _ in range(20): 
            env.core_env.step({'blocks': init_blocks, 'velocity': [0]*4, 'steering': [0]*4})
            
        time.sleep(1)
        last_cmd = np.zeros(3)
        max_steps = int(80.0 / 0.05)
        current_phi = 1.5 
        last_yaw_offset = 0.0 # 记录上一时刻偏置用于检测转体状态
        last_stable_yaw = None # 【稳像器状态位】
        
        try:
            for step in range(max_steps):
                obs = env.get_obs()
                ref_traj_10, target_phi = brain.get_action(obs, global_traj)
                
                if mode == 2:
                    target_phi = 1.5
                
                # ---------------------------------------------------------
                # 【软件级姿态稳像器 - Stabilizer V2】
                # 问题根源：Phi=3 后 Base Link 结构位置改变，其偏航角读数
                # 在机器人移动时产生巨大跳变，导致原地抽搐。
                # 解决：Phi>2.8 时改用前后轮组几何连线向量计算真实车头朝向，
                # 完全绕开 Base Link 的不稳定读数。
                # ---------------------------------------------------------
                x_w, y_w, raw_yaw = obs['pose']
                if last_stable_yaw is None:
                    last_stable_yaw = raw_yaw
                
                if current_phi > 2.8:
                    # 【Phi=3 专用 - Pivot 中垂线法】
                    # 读取 4 个 pivot 关节世界坐标，找最短距离对，
                    # 取该连线的中垂线（朝外侧）作为机器人正方向。
                    # 彻底绕开 Base Link 读数，物理上最稳健。
                    try:
                        pivot_pos = []
                        for p_id in env.core_env.pivot_ids:
                            lk = p.getLinkState(env.core_env.robotId, p_id)
                            pivot_pos.append(np.array(lk[0][:2]))
                        
                        # 遍历所有 6 对，找最短距离对
                        n = len(pivot_pos)
                        min_dist = float('inf')
                        min_pair = (0, 1)
                        for ii in range(n):
                            for jj in range(ii + 1, n):
                                d = np.linalg.norm(pivot_pos[ii] - pivot_pos[jj])
                                if d < min_dist:
                                    min_dist = d
                                    min_pair = (ii, jj)
                        
                        # 最短对连线方向 → 取垂直方向（两个候选）
                        line_vec = pivot_pos[min_pair[1]] - pivot_pos[min_pair[0]]
                        line_vec /= np.linalg.norm(line_vec) + 1e-9
                        perp1 = np.array([-line_vec[1],  line_vec[0]])
                        perp2 = np.array([ line_vec[1], -line_vec[0]])
                        
                        # 选择与上一时刻方向对齐的那个（朝外侧 = 连续方向）
                        ref_vec = np.array([np.cos(last_stable_yaw), np.sin(last_stable_yaw)])
                        forward_vec = perp1 if np.dot(perp1, ref_vec) >= np.dot(perp2, ref_vec) else perp2
                        geom_yaw = np.arctan2(forward_vec[1], forward_vec[0])
                        
                        # 轻量平滑（alpha=0.3，滤高频不引入滞后）
                        yaw_diff = (geom_yaw - last_stable_yaw + np.pi) % (2 * np.pi) - np.pi
                        stable_yaw = (last_stable_yaw + 0.3 * yaw_diff + np.pi) % (2 * np.pi) - np.pi
                    except:
                        stable_yaw = last_stable_yaw  # 保险回退
                else:
                    # 普通形态：使用低通滤波的 Base Link 偏航角
                    alpha = np.clip(1.0 - (current_phi - 2.0) * 0.45, 0.3, 1.0)
                    yaw_diff = (raw_yaw - last_stable_yaw + np.pi) % (2 * np.pi) - np.pi
                    stable_yaw = (last_stable_yaw + alpha * yaw_diff + np.pi) % (2 * np.pi) - np.pi
                
                last_stable_yaw = stable_yaw
                yaw_w = stable_yaw  # 更新可视化和控制系统的 Yaw 输入

                # 【视觉性能优化】: 每 5 帧刷新一次 Debug 绘图
                if step % 5 == 0:
                    p.removeAllUserDebugItems()
                    # 1. 绘制局部参考轨迹 (红线)
                    for k in range(0, 9, 2):
                         p.addUserDebugLine([ref_traj_10[k,0], ref_traj_10[k,1], 0.1],
                                            [ref_traj_10[k+1,0], ref_traj_10[k+1,1], 0.1], [1,0,0], 3)
                    # 2. 绘制车头箭头 (绿色 - 稳像版)
                    head_x = x_w + 0.5 * np.cos(yaw_w)
                    head_y = y_w + 0.5 * np.sin(yaw_w)
                    p.addUserDebugLine([x_w, y_w, 0.22], [head_x, head_y, 0.22], [0, 1, 0], 5)
                    # 箭头小尖头
                    p.addUserDebugLine([head_x, head_y, 0.22], 
                                       [head_x - 0.1*np.cos(yaw_w+0.4), head_y - 0.1*np.sin(yaw_w+0.4), 0.22], [0, 1, 0], 5)
 
                # ---------------------------------------------------------
                # 姿态修正策略【已注释 - 谐波误差偏置功能暂时关闭】
                # 原逻辑：在 Phi=2.2~2.8 之间线性叠加 0->90 度的转向补偿
                # 当前第三个障碍物（大桥）的偏置角原为 90 度（π/2 rad）
                # 改为 0 度正面直接通过，暂时注释掉以下偏置逻辑
                # ---------------------------------------------------------
                # phi_progress = np.clip((current_phi - 2.2) / 0.6, 0.0, 1.0)
                # yaw_offset = phi_progress * 1.5708
                # is_rotating = abs(yaw_offset - last_yaw_offset) > 0.001
                # last_yaw_offset = yaw_offset
                # ref_traj_10[:, 2] += yaw_offset
                # ref_traj_10[:, 2] = (ref_traj_10[:, 2] + np.pi) % (2 * np.pi) - np.pi
                
                # 偏置关闭后，转体过渡期标志恒为 False
                is_rotating = False

                # MPC 控制
                current_state = np.array([x_w, y_w, yaw_w])
                cmd = mpc.solve(current_state, ref_traj_10, last_cmd)
                
                # ---------------------------------------------------------
                # 越障辅助: 动态速度缩放 (回归高精度驻车方案)
                # 针对 Phi=3 或正在转体期间，执行驻车变身策略
                # ---------------------------------------------------------
                phi_error = abs(target_phi - current_phi)
                
                # 特异性逻辑：涉及到 Phi=3 的变形（进出大桥）或转体，原地静止变形
                if (target_phi > 2.8 or current_phi > 2.8 or is_rotating) and phi_error > 0.05:
                    speed_scale = 0.0
                else:
                    # 提高普通变身时的速度下限（由 0.2 提高至 0.5），保证“边走边变”的流畅度
                    speed_scale = np.clip(1.0 - phi_error * 2.0, 0.5, 1.0)
                
                vx_b, vy_b, omega = cmd[0] * speed_scale, cmd[1] * speed_scale, cmd[2] * speed_scale
                vx_world = vx_b * np.cos(yaw_w) - vy_b * np.sin(yaw_w)
                vy_world = vx_b * np.sin(yaw_w) + vy_b * np.cos(yaw_w)
                
                # 逆运动学
                raw_zero_yaws = env.core_env.get_steering_zero_yaws_imu()
                zero_yaws_w = smoother.update(raw_zero_yaws)
                for i, s_id in enumerate(env.core_env.steering_ids):
                    corner = env.core_env.STEER_CORNER_ORDER[i]
                    swerve.last_angles[corner] = p.getJointState(env.core_env.robotId, s_id)[0]
                    
                cmds_sw = swerve.compute_swerve_kinematics(vx_world, vy_world, omega, 
                                                        env.core_env.get_wheel_positions_for_kinematics()[0], zero_yaws_w)
                                                        
                steers = [cmds_sw[env.core_env.STEER_CORNER_ORDER[i]]['angle'] for i in range(4)]
                vels = [cmds_sw[env.core_env.STEER_CORNER_ORDER[i]]['speed'] for i in range(4)]
                
                # 形变控制
                max_delta_phi = 0.5 * 0.05
                if target_phi > current_phi:
                    current_phi = min(target_phi, current_phi + max_delta_phi)
                elif target_phi < current_phi:
                    current_phi = max(target_phi, current_phi - max_delta_phi)
                
                # ---------------------------------------------------------
                # 动态刚度切换: 越障全程锁定保护策略
                # ---------------------------------------------------------
                # 剧烈变形或剧烈转体时，强制进入柔性随动模式
                is_heavy_action = (target_phi > 2.8 or current_phi > 2.8 or is_rotating) and phi_error > 0.05
                
                env.core_env.dynamic_pivot_is_morphing = (phi_error > 0.005) or is_heavy_action
                
                if is_heavy_action:
                    env.core_env.dynamic_pivot_is_crossing = False
                else:
                    env.core_env.dynamic_pivot_is_crossing = (obs['closest_type'] != 'none') and (-1.8 < obs['closest_dist'] < 0.8)
                
                # 调用由脚本注入的自适应硬锁定补丁
                env.core_env.update_pivot_hold()
                
                if gnn_model is not None:
                    block_angles = gnn_predict(gnn_model, gnn_converter, current_phi, edge_index, device)
                else:
                    block_angles = compute_angles_from_phi(current_phi)
                
                val_action = {'steering': steers, 'velocity': vels, 'blocks': block_angles}
                obs, success, info = env.step(val_action)
                
                if -1.2 < obs.get('closest_dist', 99) < 2.0:
                    print(f"[Step {step}] {brain.state} | NextObs Dist: {obs.get('closest_dist', 99):.2f}m")
                
                if success:
                    print(f"\n[OK] 三层架构成功引导机器人顺利抵达终点！时间: {(step*0.05):.1f}s")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            env.core_env.close()

    # ----------------------------------------------------------------------
    # 模式 2: 场景展示模式
    # ----------------------------------------------------------------------
    elif main_menu == 2:
        print("\n--- ABC场景选择展示 (无机器人) ---")
        print(" 请选择要视察的场景编号：")
        print("  [1] Scene A (致命窄门)")
        print("  [2] Scene B (随机航点)")
        print("  [3] Scene C (终极综合全魔关卡)")
        
        try:
            s_choice = int(input("请输入场景编号 (1-3) [默认1]: ").strip())
        except:
            s_choice = 1
            
        scene_id = {1: 'A', 2: 'B', 3: 'C'}.get(s_choice, 'A')
        
        env = BenchmarkEnv(urdf_path, gui=True, scene_id=scene_id)
        env.reset()
        p.removeBody(env.core_env.robotId)
        
        traj = env.task.global_traj
        if traj is not None:
            for k in range(0, len(traj)-5, 5):
                p.addUserDebugLine([traj[k,0], traj[k,1], 0.05],
                                   [traj[k+5,0], traj[k+5,1], 0.05], 
                                   lineColorRGB=[0.1, 0.5, 1.0], lineWidth=4)
                                   
        print(f"[OK] Scene {scene_id} 铺设完毕。按 Ctrl+C 退出。")
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0) # 关闭阴影提速
        try:
            while True:
                p.stepSimulation()
                time.sleep(1./2400.) # 10 倍速推进
        except KeyboardInterrupt:
            env.core_env.close()

if __name__ == "__main__":
    main()

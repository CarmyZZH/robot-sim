import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import types
from scipy.interpolate import CubicSpline

# 确保能找到项目根目录下的包
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.envs.benchmark_env import BenchmarkEnv
from src.controllers.mpc_controller import KinematicMPC
from src.controllers.swerve_kinematics import SwerveDriveController
from scripts.test_mpc_vs_pid_topology import ZeroYawSmoother, compute_angles_from_phi
from src.brains.scripted_brain import ScriptedBrain
import torch

def load_gnn_model_env(project_root, device):
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
        print(f"[BrainGymEnv] ✓ 已加载 GNN: {os.path.basename(os.path.dirname(model_path))}")
        return model, converter
    except Exception as e:
        print(f"[BrainGymEnv] ✗ GNN 加载失败: {e}")
        return None, None

def gnn_predict_env(model, converter, phi, edge_index, device):
    features = converter.get_feature_matrix(phi)
    x = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x, edge_index)
    g = pred[0].squeeze(-1).cpu().numpy()  # model returns (final_output, raw_output)
    return converter.graph_to_env_angles(g)

class BrainGymEnv(gym.Env):
    """
    符合 Gymnasium 标准的 DRL 环境包装器。
    运行频率: 1Hz (每次 step 执行 1.0s)
    内部物理与 MPC 频率: 20Hz (每次 step 内部循环 20 次 0.05s)
    """
    def __init__(self, urdf_path, gui=False, scene_id='C', max_steps=100,
                 imitation_weight=1.0, expert_lookahead_idx=5):
        super(BrainGymEnv, self).__init__()
        self.gui = gui
        self.scene_id = scene_id
        self.max_steps = max_steps
        self.imitation_weight = imitation_weight
        self.expert_lookahead_idx = expert_lookahead_idx
        
        self.env = BenchmarkEnv(urdf_path, gui=gui, scene_id=scene_id)
        
        # 动作空间: 3D 连续空间, 范围 [-1, 1]
        # a[0]: dx_end_norm, 映射到局部坐标 X 轴的位移 (前后)
        # a[1]: dy_end_norm, 映射到局部坐标 Y 轴的位移 (左右)
        # a[2]: phi_target_norm, 映射到 [0, 3] 的形变目标
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # 观测空间: 22D 语义雷达
        # [rx, ry, ryaw, gx, gy, phi_curr, dist_to_goal, obs1_dx, obs1_dy, obs1_w, type1_1, type1_2, type1_3, type1_4, obs2...]
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(21,), dtype=np.float32)
        
        self.mpc = KinematicMPC(dt=0.05, N=10)
        self.swerve = SwerveDriveController(wheel_radius=self.env.core_env.WHEEL_RADIUS)
        self.smoother = ZeroYawSmoother(alpha=0.9)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn_model, self.gnn_converter = load_gnn_model_env(project_root, self.device)
        self.edge_index = self.gnn_converter.edge_index_7.to(self.device) if self.gnn_model is not None else None
        
        # 注入硬锁定补丁
        self._inject_pivot_hold_patch()
        
        self.current_step = 0
        self.current_phi = 1.5
        self.last_stable_yaw = None
        self.last_cmd = np.zeros(3)
        self.last_stable_yaw = 0.0
        
        # 实例化专家导师 (ScriptedBrain)，用于引导训练
        self.teacher = ScriptedBrain()

    def _inject_pivot_hold_patch(self):
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
                    curr_pos = p.getJointState(self_env.robotId, p_id)[0]
                    p.setJointMotorControl2(self_env.robotId, p_id, p.POSITION_CONTROL, targetPosition=curr_pos, force=0.5)
                else:
                    target = self_env.pivot_anchors[i]
                    p.setJointMotorControl2(self_env.robotId, p_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
                    p.resetJointState(self_env.robotId, p_id, targetValue=target, targetVelocity=0)
                    
        self.env.core_env.update_pivot_hold = types.MethodType(adaptive_pivot_hold, self.env.core_env)
        self.env.core_env.dynamic_pivot_is_crossing = False
        self.env.core_env.pivot_anchors = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        
        # 等待物理引擎稳定
        if self.gnn_model is not None:
            init_blocks = gnn_predict_env(self.gnn_model, self.gnn_converter, 1.5, self.edge_index, self.device)
        else:
            init_blocks = compute_angles_from_phi(1.5)
            
        for _ in range(20):
            self.env.core_env.step({'blocks': init_blocks, 'velocity': [0]*4, 'steering': [0]*4})
            
        self.current_step = 0
        self.current_phi = 1.5
        self.last_stable_yaw = None
        self.last_cmd = np.zeros(3)
        self.last_stable_yaw = 0.0
        self.last_dist_to_goal = np.hypot(self.env.task.start_pose[0] - self.env.task.goal_pose[0],
                                          self.env.task.start_pose[1] - self.env.task.goal_pose[1])
        
        # 每个 episode 重建专家，防止跨 episode 状态泄漏
        self.teacher = ScriptedBrain()
        
        return self._get_semantic_obs(), {}

    def _get_semantic_obs(self):
        base_obs = self.env.get_obs()
        rx, ry, raw_yaw = base_obs['pose']
        gx, gy, _ = self.env.task.goal_pose
        
        # --- 稳像器逻辑 ---
        stable_yaw = raw_yaw
        if self.current_phi > 2.8:
            try:
                pivot_pos = []
                for p_id in self.env.core_env.pivot_ids:
                    lk = p.getLinkState(self.env.core_env.robotId, p_id)
                    pivot_pos.append(np.array(lk[0][:2]))
                
                n = len(pivot_pos)
                min_dist = float('inf')
                min_pair = (0, 1)
                for ii in range(n):
                    for jj in range(ii + 1, n):
                        d = np.linalg.norm(pivot_pos[ii] - pivot_pos[jj])
                        if d < min_dist:
                            min_dist = d
                            min_pair = (ii, jj)
                
                line_vec = pivot_pos[min_pair[1]] - pivot_pos[min_pair[0]]
                line_vec /= np.linalg.norm(line_vec) + 1e-9
                perp1 = np.array([-line_vec[1],  line_vec[0]])
                perp2 = np.array([ line_vec[1], -line_vec[0]])
                
                if self.last_stable_yaw is None:
                    self.last_stable_yaw = raw_yaw
                ref_vec = np.array([np.cos(self.last_stable_yaw), np.sin(self.last_stable_yaw)])
                forward_vec = perp1 if np.dot(perp1, ref_vec) >= np.dot(perp2, ref_vec) else perp2
                geom_yaw = np.arctan2(forward_vec[1], forward_vec[0])
                
                yaw_diff = (geom_yaw - self.last_stable_yaw + np.pi) % (2 * np.pi) - np.pi
                stable_yaw = (self.last_stable_yaw + 0.3 * yaw_diff + np.pi) % (2 * np.pi) - np.pi
            except:
                stable_yaw = self.last_stable_yaw if self.last_stable_yaw is not None else raw_yaw
        else:
            if self.last_stable_yaw is None:
                self.last_stable_yaw = raw_yaw
            alpha = np.clip(1.0 - (self.current_phi - 2.0) * 0.45, 0.3, 1.0)
            yaw_diff = (raw_yaw - self.last_stable_yaw + np.pi) % (2 * np.pi) - np.pi
            stable_yaw = (self.last_stable_yaw + alpha * yaw_diff + np.pi) % (2 * np.pi) - np.pi
            
        self.last_stable_yaw = stable_yaw
        
        dist_to_goal = np.hypot(rx - gx, ry - gy)
        
        # 构建 7D 自身状态
        state = [rx, ry, stable_yaw, gx, gy, self.current_phi, dist_to_goal]
        
        # --- 语义雷达 (感知最近的 K=2 障碍物) ---
        # 我们遍历场景中存在的障碍物并计算与机器人的距离
        obstacles_info = []
        
        # 预先计算用于坐标系转换的三角函数
        cos_y = np.cos(stable_yaw)
        sin_y = np.sin(stable_yaw)
        
        if hasattr(self.env.task, 'boss_obstacles'):
            for obs_ in self.env.task.boss_obstacles:
                t = obs_['type']
                ox, oy = obs_['x'], obs_.get('y', 0.0)
                
                dx_world = ox - rx
                dy_world = oy - ry
                dist = np.hypot(dx_world, dy_world)
                
                # 坐标系转换：相对机器人局部坐标系 (X为正前方，Y为左正右负)
                dx_local = dx_world * cos_y + dy_world * sin_y
                dy_local = -dx_world * sin_y + dy_world * cos_y
                
                width = obs_.get('width', 0.2)
                if t == 'narrow_gate': width = obs_.get('width', 0.15)
                elif t == 'cylinders': width = obs_.get('radius', 0.2) * 2.0
                
                # 仅考虑前方/附近的障碍物 (简化：局部前方 5m 内，允许在身后 1m 以防刚穿过就被丢弃)
                if dist < 5.0 and dx_local > -1.0: 
                    type_onehot = [
                        1.0 if t == 'narrow_gate' else 0.0,
                        1.0 if t == 'low_platform' else 0.0,
                        1.0 if t == 'bridge' else 0.0,
                        1.0 if t == 'cylinders' else 0.0
                    ]
                    obstacles_info.append({
                        'dist': dist,
                        'data': [dx_local, dy_local, width] + type_onehot
                    })
        
        # 兜底：窄门单门情况
        elif self.env.task.door_config is not None:
            dc = self.env.task.door_config
            ox = dc['x']
            oy = dc.get('y_center', 0.0)
            dx_world = ox - rx
            dy_world = oy - ry
            dist = np.hypot(dx_world, dy_world)
            
            dx_local = dx_world * cos_y + dy_world * sin_y
            dy_local = -dx_world * sin_y + dy_world * cos_y
            
            if dist < 5.0 and dx_local > -1.0:
                obstacles_info.append({
                    'dist': dist,
                    'data': [dx_local, dy_local, dc['width'], 1.0, 0.0, 0.0, 0.0]
                })

        # 按距离排序
        obstacles_info.sort(key=lambda x: x['dist'])
        
        # 提取前两个
        obs_k = []
        for i in range(2):
            if i < len(obstacles_info):
                obs_k.extend(obstacles_info[i]['data'])
            else:
                obs_k.extend([0.0]*7) # 填 0
                
        final_obs = np.array(state + obs_k, dtype=np.float32)
        return final_obs

    def _generate_trajectory(self, dx_local, dy_local):
        """用平滑曲线连接当前点到预测目标点"""
        rx, ry, ryaw = self.last_stable_yaw_pose # 格式 (x, y, yaw)
        
        # 映射 DRL 输出 [-1, 1] 到实际最大预测距离 (例如最大 1.0m)
        dx_local *= 1.0
        dy_local *= 1.0
        
        # 转世界坐标
        dx_w = dx_local * np.cos(ryaw) - dy_local * np.sin(ryaw)
        dy_w = dx_local * np.sin(ryaw) + dy_local * np.cos(ryaw)
        target_x = rx + dx_w
        target_y = ry + dy_w
        
        # 生成 10 步插值点
        t_pts = np.linspace(0, 1, 10)
        
        # 简单线性插值 (可改为样条曲线，但由于是局部 1m 以内，线性足以，且计算快)
        traj_x = np.linspace(rx, target_x, 10)
        traj_y = np.linspace(ry, target_y, 10)
        traj_yaw = np.ones(10) * ryaw # 简化方向
        
        ref_traj = np.vstack((traj_x, traj_y, traj_yaw)).T
        return ref_traj

    def step(self, action):
        # 1. 解析 Action
        dx_local = action[0]
        dy_local = action[1]
        target_phi = 1.5 * action[2] + 1.5 # [-1, 1] -> [0, 3]
        
        # 拿到观测时的物理状态
        base_obs = self.env.get_obs()
        self.last_stable_yaw_pose = (base_obs['pose'][0], base_obs['pose'][1], self.last_stable_yaw)
        
        # 2. 生成 10 步参考轨迹 (1s 内)
        ref_traj_10 = self._generate_trajectory(dx_local, dy_local)
        
        # 3. 开始 20 步的 MPC 高频闭环执行 (20 * 0.05s = 1.0s)
        accumulated_reward = 0.0
        done = False
        info = {}
        
        for k in range(20):
            # 每 2 步消耗一个 trajectory point
            traj_idx = min(k // 2, 9)
            
            # 使用剩余的部分轨迹
            curr_ref = ref_traj_10[traj_idx:]
            
            # 读取当前状态
            obs_dict = self.env.get_obs()
            rx, ry, raw_yaw = obs_dict['pose']
            stable_yaw = self.last_stable_yaw # 复用平滑结果
            yaw_w = stable_yaw
            
            current_state = np.array([rx, ry, yaw_w])
            
            # MPC 控制
            cmd = self.mpc.solve(current_state, curr_ref, self.last_cmd)
            self.last_cmd = cmd
            
            # 变形和平滑调速
            phi_error = abs(target_phi - self.current_phi)
            
            is_rotating = False
            is_heavy_action = (target_phi > 2.8 or self.current_phi > 2.8 or is_rotating) and phi_error > 0.05
            if is_heavy_action:
                speed_scale = 0.0
            else:
                speed_scale = np.clip(1.0 - phi_error * 2.0, 0.5, 1.0)
                
            vx_b, vy_b, omega = cmd[0] * speed_scale, cmd[1] * speed_scale, cmd[2] * speed_scale
            vx_world = vx_b * np.cos(yaw_w) - vy_b * np.sin(yaw_w)
            vy_world = vx_b * np.sin(yaw_w) + vy_b * np.cos(yaw_w)
            
            # 运动学解算
            raw_zero_yaws = self.env.core_env.get_steering_zero_yaws_imu()
            zero_yaws_w = self.smoother.update(raw_zero_yaws)
            
            for i, s_id in enumerate(self.env.core_env.steering_ids):
                corner = self.env.core_env.STEER_CORNER_ORDER[i]
                self.swerve.last_angles[corner] = p.getJointState(self.env.core_env.robotId, s_id)[0]
                
            cmds_sw = self.swerve.compute_swerve_kinematics(vx_world, vy_world, omega, 
                                                            self.env.core_env.get_wheel_positions_for_kinematics()[0], zero_yaws_w)
                                                            
            steers = [cmds_sw[self.env.core_env.STEER_CORNER_ORDER[i]]['angle'] for i in range(4)]
            vels = [cmds_sw[self.env.core_env.STEER_CORNER_ORDER[i]]['speed'] for i in range(4)]
            
            # Phi 步进
            max_delta_phi = 0.5 * 0.05
            if target_phi > self.current_phi:
                self.current_phi = min(target_phi, self.current_phi + max_delta_phi)
            elif target_phi < self.current_phi:
                self.current_phi = max(target_phi, self.current_phi - max_delta_phi)
                
            # 锁定保护
            self.env.core_env.dynamic_pivot_is_morphing = (phi_error > 0.005) or is_heavy_action
            if is_heavy_action:
                self.env.core_env.dynamic_pivot_is_crossing = False
            else:
                self.env.core_env.dynamic_pivot_is_crossing = (obs_dict['closest_type'] != 'none') and (-1.8 < obs_dict['closest_dist'] < 0.8)
            
            self.env.core_env.update_pivot_hold()
            
            if self.gnn_model is not None:
                block_angles = gnn_predict_env(self.gnn_model, self.gnn_converter, self.current_phi, self.edge_index, self.device)
            else:
                block_angles = compute_angles_from_phi(self.current_phi)
                
            val_action = {'steering': steers, 'velocity': vels, 'blocks': block_angles}
            _, success, _ = self.env.step(val_action)
            
            # 检测碰撞 (简单基于底盘接触，或高度过低)
            # 在此简化处理，实际可以依靠 PyBullet 的 getContactPoints
            
            if success:
                done = True
                accumulated_reward += 100.0 # 成功巨额奖励
                break
                
        # 4. 计算大步 Reward
        self.current_step += 1
        
        # 奖励设计
        # 1. 进展奖励
        curr_dist_to_goal = np.hypot(rx - self.env.task.goal_pose[0], ry - self.env.task.goal_pose[1])
        progress = self.last_dist_to_goal - curr_dist_to_goal
        accumulated_reward += progress * 10.0 # 前进 1 米给 10 分
        self.last_dist_to_goal = curr_dist_to_goal
        
        # 2. 时间惩罚
        accumulated_reward -= 0.1 
        
        # 3. 中线偏离惩罚 (防止绕路逃课)
        accumulated_reward -= abs(ry) * 0.02
        
        # 4. 无效变形惩罚
        if abs(phi_error) > 0.1:
            accumulated_reward -= 0.05
            
        # 5. 老师引导 (Teacher Forcing Reward)
        # 根据规则大脑的经验，在障碍物附近提供形态的梯度奖励，加速收敛
        final_obs = self._get_semantic_obs() # 修复：获取最新的语义观测数据
        obs_dict = self.env.get_obs()
        c_type = obs_dict['closest_type']
        c_dist = obs_dict['closest_dist']
        
        # 寻找最近障碍物的 dy_local (在 final_obs 的对应位置)
        # 根据之前的代码，obs_k 的数据结构是 [dx, dy, width, t1, t2, t3, t4]
        # 它在 final_obs 中的索引是 state_len (7) + 1 = 8
        dy_local_obs = final_obs[8] if len(final_obs) > 8 else 0.0

        # 当距离障碍物较近（[-0.5m, 2.5m]），且机器人在前进时，进行形态考核
        if c_type != 'none' and -0.5 < c_dist < 2.5:
            # --- 形态考核 (原有的逻辑) ---
            optimal_phi = 1.5
            if c_type == 'narrow_gate': optimal_phi = 0.0
            elif c_type == 'low_platform': optimal_phi = 2.0
            elif c_type == 'bridge': optimal_phi = 3.0
            elif c_type == 'cylinders': optimal_phi = 1.5
            
            phi_diff = abs(self.current_phi - optimal_phi)
            shape_multiplier = max((0.5 - phi_diff) * 2.0, -1.0)
            hint_reward = (progress * 10.0) * shape_multiplier
            accumulated_reward += hint_reward

            # --- 新增：对准引导 (Alignment Reward) ---
            # 针对窄门和大桥，额外奖励 Y 轴对齐 (dy_local 越近 0 越好)
            if c_type in ['narrow_gate', 'bridge'] and progress > 0:
                align_reward = np.exp(-5.0 * abs(dy_local_obs)) * 1.0 # 满分 +1.0
                accumulated_reward += align_reward

        # 6. 碰撞检测与“撞墙即死”策略
        # 优化：在空旷地带给予更多探索空间，仅在障碍物附近或长期静止时触发判定
        near_obstacle = (c_type != 'none' and c_dist < 1.0)
        
        # 判定条件：
        # A. 在障碍物附近卡住 (progress < 2mm)
        # B. 即使不在障碍物附近，但已经过了 15 步还没怎么动弹 (排除起步纠结期)
        is_stuck = (progress < 0.002) and (near_obstacle or self.current_step > 15)
        
        if is_stuck:
            accumulated_reward -= 20.0 # 强化重罚，从 -5 提升至 -20
            done = True 
            # print(f"💥 Collision! Final Penalty: -20. Restarting...")

        # 7. 专家动作对齐奖励 (Expert Action Alignment) - 课程学习加速手段
        # imitation_weight 可在训练过程中从 1.0 衰减到 0，实现从模仿到自主的过渡
        if not done and self.teacher is not None and self.imitation_weight > 0.0:
            # 获取当前物理状态下的专家动作
            expert_ref_traj, expert_phi = self.teacher.get_action(obs_dict, self.env.task.global_traj)

            # 使用前瞻点而非第 0 点，避免目标过近导致信号坍缩
            idx = min(self.expert_lookahead_idx, len(expert_ref_traj) - 1)
            ex_world_dx = expert_ref_traj[idx][0] - rx
            ex_world_dy = expert_ref_traj[idx][1] - ry

            # 转换到机器人局部坐标系，与 DRL 动作空间对齐
            cos_y = np.cos(self.last_stable_yaw)
            sin_y = np.sin(self.last_stable_yaw)
            ex_dx_local = ex_world_dx * cos_y + ex_world_dy * sin_y
            ex_dy_local = -ex_world_dx * sin_y + ex_world_dy * cos_y

            ex_phi_norm = (expert_phi - 1.5) / 1.5

            expert_action = np.array([
                np.clip(ex_dx_local, -1.0, 1.0),
                np.clip(ex_dy_local, -1.0, 1.0),
                np.clip(ex_phi_norm, -1.0, 1.0),
            ], dtype=np.float32)

            # ── 方向余弦奖励（独立于 progress，直接奖励朝专家方向走）──
            # 计算 PPO 导航动作与专家导航方向的余弦相似度
            ppo_nav = action[:2]                        # (dx, dy) 归一化后的动作
            ex_nav  = expert_action[:2]
            ppo_norm = np.linalg.norm(ppo_nav) + 1e-8
            ex_norm  = np.linalg.norm(ex_nav)  + 1e-8
            direction_cos = np.dot(ppo_nav, ex_nav) / (ppo_norm * ex_norm)  # [-1, 1]
            # 乘以 3.0 使其量级与 progress*10 相当（前进0.1m才有1分，方向对就给3分）
            direction_reward = 3.0 * direction_cos
            accumulated_reward += self.imitation_weight * direction_reward

            # ── Φ 对齐奖励（保持形态关联学习）──
            phi_diff = abs(action[2] - expert_action[2])
            phi_score = np.clip((0.4 - phi_diff) * 2.5, -1.0, 1.0)
            accumulated_reward += self.imitation_weight * phi_score

        if self.current_step >= self.max_steps:
            done = True
            
        next_obs = self._get_semantic_obs()
        return next_obs, accumulated_reward, done, False, info

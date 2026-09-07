import os
import pybullet as p
import numpy as np
from envs.robot_env import MorphingRobotEnv
from envs.task_manager import BenchmarkTask

class BenchmarkEnv:
    """包装底壳，提供统一的标准沙盘对抗平台"""
    def __init__(self, urdf_path, gui=True, scene_id='A', add_boundaries=True):
        # 兼容原本优秀的物理模型
        self.core_env = MorphingRobotEnv(urdf_path, gui=gui, control_steps=12)
        self.task = BenchmarkTask(scene_id=scene_id, dt=0.05)
        self.add_boundaries = add_boundaries
        self.walls = []
        print(f"[OK] Scene {scene_id} 铺设完毕。按 Ctrl+C 退出。")
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0) # 关闭阴影提速
        
    def reset(self):
        self.core_env.reset()
        
        # 打扫上一场残留的障碍物
        for w in self.walls:
            try:
                p.removeBody(w)
            except: pass
        self.walls = []
        
        # 根据当前选择的场景装修沙箱地图
        self._decorate_scene()
            
        return self.get_obs()
        
    def _decorate_scene(self):
        if self.task.door_config is not None:
            # 搭建 Scene A 的变态窄门
            dc = self.task.door_config
            gate_x = dc['x']
            gap = dc['width'] # 缝隙宽度 0.15
            thickness = dc['thickness']
            
            # 使用 PyBullet 高速生成静态刚体墙壁
            # 左墙 (在 Y 轴负半侧)
            col_id_l = p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness/2, 2.0, 0.5])
            vis_id_l = p.createVisualShape(p.GEOM_BOX, halfExtents=[thickness/2, 2.0, 0.5], rgbaColor=[0.5, 0.5, 0.5, 1])
            wall_l = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id_l, baseVisualShapeIndex=vis_id_l,
                                       basePosition=[gate_x, -2.0 - gap/2, 0.5])
                                       
            # 右墙 (在 Y 轴正半侧)
            col_id_r = p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness/2, 2.0, 0.5])
            vis_id_r = p.createVisualShape(p.GEOM_BOX, halfExtents=[thickness/2, 2.0, 0.5], rgbaColor=[0.5, 0.5, 0.5, 1])
            wall_r = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id_r, baseVisualShapeIndex=vis_id_r,
                                       basePosition=[gate_x, 2.0 + gap/2, 0.5])
                                       
            self.walls.extend([wall_l, wall_r])
            
        # 搭建 Scene C 的终极复合地狱障碍！
        if hasattr(self.task, 'boss_obstacles'):
            for obs in self.task.boss_obstacles:
                t, x, y = obs['type'], obs['x'], obs['y']
                
                if t == 'narrow_gate':
                    gap = obs['width'] 
                    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 2.0, 0.5])
                    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 2.0, 0.5], rgbaColor=[0.5, 0.5, 0.5, 1])
                    w1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[x, y - 2.0 - gap/2, 0.5])
                    w2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[x, y + 2.0 + gap/2, 0.5])
                    self.walls.extend([w1, w2])
                    
                elif t == 'low_platform':
                    w, h = obs['width'], obs['height']
                    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, w/2, h/2])
                    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, w/2, h/2], rgbaColor=[0.8, 0.6, 0.1, 1])
                    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[x, y, h/2])
                    self.walls.append(body)
                    
                elif t == 'bridge':
                    w, h = obs['width'], obs['height']
                    # 顶盖 (压制底盘)
                    r_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 1.5, 0.1])
                    r_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 1.5, 0.1], rgbaColor=[0.2, 0.8, 0.8, 1])
                    body_roof = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=r_col, baseVisualShapeIndex=r_vis, basePosition=[x, y, h + 0.1])
                    # 两侧厚实立柱
                    p_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.6, h/2])
                    p_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.6, h/2], rgbaColor=[0.2, 0.8, 0.8, 1])
                    body_l = p.createMultiBody(0, p_col, p_vis, [x, y - (w/2 + 0.6), h/2])
                    body_r = p.createMultiBody(0, p_col, p_vis, [x, y + (w/2 + 0.6), h/2])
                    self.walls.extend([body_roof, body_l, body_r])
                    
                elif t == 'cylinders':
                    r = obs['radius']
                    col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=0.6)
                    vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=0.6, rgbaColor=[0.8, 0.2, 0.2, 1])
                    # 【用户最新要求】: 两个柱子左右并排，净间距 1.0m (中心距 1.4m)
                    body1 = p.createMultiBody(0, col_id, vis_id, [x, y + 0.7, 0.3])
                    body2 = p.createMultiBody(0, col_id, vis_id, [x, y - 0.7, 0.3])
                    self.walls.extend([body1, body2])
                
        # 搭建 Scene B 的空中虚拟路标 (仅限视觉可见，无物理碰撞的绿色航点)
        if hasattr(self.task, 'waypoints') and self.task.waypoints is not None:
            for i, wp in enumerate(self.task.waypoints):
                vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[0.1, 0.9, 0.1, 0.7])
                # 无碰撞 id 意味着它是幻影
                body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_id,
                                         basePosition=[wp[0], wp[1], 0.3])
                self.walls.append(body)
                
        # 搭建全场景统一的“空气墙”赛道边界，防止 DRL 逃课绕远路
        if self.add_boundaries:
            track_length = 30.0
            track_width = 3.0  # y=[-3, 3]
            col_bound = p.createCollisionShape(p.GEOM_BOX, halfExtents=[track_length/2, 0.1, 1.0])
            vis_bound = p.createVisualShape(p.GEOM_BOX, halfExtents=[track_length/2, 0.1, 1.0], rgbaColor=[1, 1, 1, 0.2])
            
            # 左边界 (y < 0)
            bound_l = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_bound, baseVisualShapeIndex=vis_bound, basePosition=[track_length/2 - 2.0, -track_width, 1.0])
            # 右边界 (y > 0)
            bound_r = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_bound, baseVisualShapeIndex=vis_bound, basePosition=[track_length/2 - 2.0, track_width, 1.0])
            
            self.walls.extend([bound_l, bound_r])
            
    def step(self, action_dict):
        """action_dict: 应该是一整套包括舵轮 velocity 等的数据（中层脑已算好的）"""
        self.core_env.step(action_dict)
        obs = self.get_obs()
        info = self.task.evaluate_state(obs['pose'])
        return obs, info['success'], info

    def get_obs(self):
        """给上层大脑看的传感器大礼包 - 核心：全部参考几何中心 (Centroid)"""
        # 1. 计算四轮支撑域的几何中心 (World X, Y)
        # 不要相信 URDF 的 Base Link，它在变形时会偏移！
        wheel_world_offsets = self.core_env.get_wheel_positions_world_frame()
        # 注意：core_env 提供的是【相对于中心】的偏量，为了拿到绝对中心，我们需要反向计算
        # 或者更直接：从 LinkState 拿到四个铰点的绝对世界坐标，然后求平均
        raw_centers = []
        for s_id in self.core_env.steering_ids:
            st = p.getLinkState(self.core_env.robotId, s_id)
            raw_centers.append(st[0]) # 世界绝对坐标
            
        cX = sum(p[0] for p in raw_centers) / 4.0
        cY = sum(p[1] for p in raw_centers) / 4.0
        yaw = self.core_env.get_robot_yaw() # Yaw 依然使用主塔姿态 (姿态是全域一致的)
        
        # 2. 基于几何中心进行“边界感知”雷达扫描
        closest_type = 'none'
        closest_dist = 999.0
        
        if hasattr(self.task, 'boss_obstacles') and self.task.boss_obstacles:
            for obs_ in self.task.boss_obstacles:
                t = obs_['type']
                half_thickness = 0.0
                if t == 'narrow_gate': half_thickness = 0.1
                elif t == 'low_platform': half_thickness = 0.25
                elif t == 'bridge': half_thickness = 0.2
                elif t == 'cylinders': half_thickness = obs_['radius']
                
                edge_x = obs_['x'] - half_thickness
                exit_x = obs_['x'] + half_thickness
                d = edge_x - cX
                
                # 【用户最新要求】: 延长至 1.0m 以防地台还没通过就变身导致碰撞
                if cX > exit_x + 1.0:
                    continue
                
                # 在剩下的生存期内的障碍物中，选择物理边缘最近的一个
                if abs(d) < abs(closest_dist):
                    closest_dist = d
                    closest_type = t
                        
        elif self.task.door_config is not None:
            edge_x = self.task.door_config['x'] - 0.1 
            if edge_x > cX - 0.8:
                closest_dist = edge_x - cX
                closest_type = 'narrow_gate'
            
        return {
            'pose': (cX, cY, yaw),
            'closest_dist': closest_dist,
            'closest_type': closest_type
        }

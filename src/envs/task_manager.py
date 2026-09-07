import numpy as np

class BenchmarkTask:
    """统一评估基准任务管家"""
    def __init__(self, scene_id='A', dt=0.05):
        self.scene_id = scene_id
        self.dt = dt
        
        self.global_traj = None
        self.obstacles = []
        self.door_config = None
        self.boss_obstacles = []
        
        self.start_pose = (0.0, 0.0, 0.0) # x, y, yaw
        self.goal_pose = (0.0, 0.0, 0.0)
        
        self._build_scene()
        
    def _build_scene(self):
        steps = int(40.0 / self.dt) # 默认规划 40 秒的跑道
        t = np.linspace(0, 40.0, steps)
        
        if self.scene_id == 'A':
            # Scene A: 窄门穿越测试
            self.start_pose = (0.0, 0.0, 0.0)
            self.goal_pose = (4.0, 0.0, 0.0)
            self.door_config = {'x': 2.0, 'y_center': 0.0, 'width': 0.42, 'thickness': 0.2}
            
            x = 0.4 * t
            y = np.zeros_like(x)
            yaw = np.zeros_like(x)
            self.global_traj = np.vstack((x, y, yaw)).T
            
        elif self.scene_id == 'B':
            # Scene B: 多目标巡航 (生成 5 个完全随机的散落目标点)
            self.start_pose = (0.0, 0.0, 0.0)
            wps_x = [0.0]
            wps_y = [0.0]
            for _ in range(5):
                wps_x.append(wps_x[-1] + np.random.uniform(0.6, 1.25))
                wps_y.append(np.random.uniform(-2.5, 2.5))
                
            self.waypoints = np.array(list(zip(wps_x[1:], wps_y[1:]))) 
            self.goal_pose = (wps_x[-1], wps_y[-1], 0.0)
            
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(wps_x, wps_y)
            total_dist = wps_x[-1]
            num_steps = int(total_dist / 0.4 / self.dt)
            xs = np.linspace(0, wps_x[-1], num_steps)
            ys = cs(xs)
            dx, dy = np.gradient(xs), np.gradient(ys)
            yaw = np.arctan2(dy, dx)
            self.global_traj = np.vstack((xs, ys, yaw)).T
            
        elif self.scene_id == 'C':
            # Scene C: 终极复合型恶魔赛道 (Boss Course) - V3: 纯直线段分段导引版
            self.start_pose = (0.0, 0.0, 0.0)
            np.random.seed(42) 
            
            boss_x_centers = [2.0, 5.0, 8.0, 11.0] # 扩大 1.0m 间距版本
            corridor_pre = 1.2  # 提前 1.2 米拉直
            corridor_post = 0.5 
            
            nodes_x = [0.0]
            nodes_y = [0.0]
            
            boss_y_values = []
            for bx in boss_x_centers:
                by = np.random.uniform(-1.2, 1.2) # 增加一点横向扰动难度
                boss_y_values.append(by)
                nodes_x.append(bx - corridor_pre)
                nodes_y.append(by)
                nodes_x.append(bx + corridor_post)
                nodes_y.append(by)
            
            nodes_x.append(nodes_x[-1] + 1.0)
            nodes_y.append(nodes_y[-1])
            self.goal_pose = (nodes_x[-1], nodes_y[-1], 0.0)
            
            # --- 提前定义障碍物列表，防止后处理 AttributeError ---
            # --- 自动对齐的障碍物定义，彻底解决 X 坐标不同步问题 ---
            self.boss_obstacles = [
                {'type': 'narrow_gate',   'x': boss_x_centers[0],  'y': boss_y_values[0],  'width': 0.42},
                {'type': 'low_platform',  'x': boss_x_centers[1],  'y': boss_y_values[1],  'width': 0.2, 'height': 0.2},
                {'type': 'bridge',        'x': boss_x_centers[2],  'y': boss_y_values[2],  'width': 0.4, 'height': 0.48},
                {'type': 'cylinders',     'x': boss_x_centers[3],  'y': boss_y_values[3],  'radius': 0.2}
            ]
            
            total_dist = nodes_x[-1]
            num_steps = int(total_dist / 0.4 / self.dt)
            xs = np.linspace(0, total_dist, num_steps)
            ys = np.interp(xs, nodes_x, nodes_y)
            
            yaw = np.zeros_like(xs)
            for i in range(len(xs) - 1):
                idx_x, idx_y = xs[i], ys[i]
                next_x, next_y = xs[i+1], ys[i+1]
                yaw[i] = np.arctan2(next_y - idx_y, next_x - idx_x)
            yaw[-1] = yaw[-2]
            
            # 暴力对齐走廊
            for i, x_val in enumerate(xs):
                for b_idx, bx in enumerate(boss_x_centers):
                    if (bx - corridor_pre - 0.02) <= x_val <= (bx + corridor_post + 0.02):
                        if self.boss_obstacles[b_idx]['type'] == 'low_platform':
                            yaw[i] = np.pi / 2.0
                        else:
                            yaw[i] = 0.0
                        break
            
            self.global_traj = np.vstack((xs, ys, yaw)).T
            
        else:
            raise ValueError(f"Unknown Scene ID: {self.scene_id}")

    def evaluate_state(self, current_pose):
        dist_to_goal = np.hypot(current_pose[0] - self.goal_pose[0], current_pose[1] - self.goal_pose[1])
        success = dist_to_goal < 0.2
        return {'dist_to_goal': dist_to_goal, 'success': success}

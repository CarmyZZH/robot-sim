import numpy as np

class BaseBrain:
    """上层大脑抽象基类 Cerebrum Layer"""
    def __init__(self, dt=0.05, mpc_horizon=10):
        self.dt = dt
        self.mpc_horizon = mpc_horizon
        self.last_closest_idx = 0

    def get_action(self, obs, global_traj):
        """
        核心思考接口:
        Args:
            obs: 包含状态信息的字典 (视需求可含 'x', 'y', 'yaw', 'phi', 'lidar_dist' 等)
            global_traj: 场景总体的引导路线 np.array (L, 3)
            
        Returns:
            ref_traj_10: numpy array (10, 3) 精准切片，完美投喂给下层 KinematicMPC
            target_phi: 期望达到的物理形变目标值 (例如 0.0=I形态，1.5=O形态)
        """
        raise NotImplementedError("所有的子脑必须手写此神经元逻辑")

    def _slice_mpc_trajectory(self, current_pose, global_traj):
        """
        内置神棍切片器: 从数百米的全局宏观路径中，切分出属于未来 0.5s (10步) 的微观路况。
        完美保障中层小脑 (Cerebellum) 毫无违和感地稳定运转。
        """
        x, y = current_pose[0], current_pose[1]
        
        # 寻找车体目前所在的最接近锚点
        search_range = min(len(global_traj), self.last_closest_idx + 100)
        dists = np.hypot(global_traj[self.last_closest_idx:search_range, 0] - x, 
                         global_traj[self.last_closest_idx:search_range, 1] - y)
        
        if len(dists) == 0:
            idx = self.last_closest_idx
        else:
            idx = self.last_closest_idx + np.argmin(dists)
            
        self.last_closest_idx = idx
        
        # 向前切分出恰好 N 步的路线
        sliced = global_traj[idx : idx + self.mpc_horizon]
        
        # 兜底：如果走到路的尽头，补充原地的 pad 数据防止数组炸裂
        if len(sliced) < self.mpc_horizon:
            if len(sliced) > 0:
                pad = np.tile(global_traj[-1], (self.mpc_horizon - len(sliced), 1))
                sliced = np.vstack([sliced, pad])
            else:
                sliced = np.tile(global_traj[-1], (self.mpc_horizon, 1))
                
        return sliced

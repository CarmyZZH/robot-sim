import pybullet as p
import numpy as np
from brains.base_brain import BaseBrain

class ScriptedBrain(BaseBrain):
    """
    第一代规则大脑：基于硬编码法则的启发式状态机 (Heuristic FSM)
    它将展现如何仅基于雷达避障距离进行宏观判定，并驱动底层做出形态收缩动作。
    """
    def __init__(self, dt=0.05, mpc_horizon=10):
        super().__init__(dt, mpc_horizon)
        self.state = 'CRUISE' # 统一状态名为巡航态
        # 默认形态：防御稳固的大圆盘 O-Shape
        self.target_phi = 1.5 
        
    def get_action(self, obs, global_traj):
        x, y, yaw = obs['pose']
        c_dist = obs.get('closest_dist', 999.0)
        c_type = obs.get('closest_type', 'none')
        
        # 【接力机制】: 如果雷达锁定的目标变了，强行重置状态机准备迎接新挑战
        if c_type != 'none' and c_type != getattr(self, 'curr_obs_type', 'none'):
            if self.state != 'CRUISE':
                print(f"[Brain] Target switch: {getattr(self, 'curr_obs_type', 'none')} -> {c_type}. Resetting FSM.")
                self.state = 'CRUISE'
            self.curr_obs_type = c_type # 更新当前目标记忆
        
        # ========== FSM (有限状态机) 大脑核心思考回路 ==========
        if self.state == 'CRUISE':
            # 缩短整体探测距离，减少在空旷带的额外形变时间
            detect_dist = 1.6
            if c_type == 'bridge': detect_dist = 1.8 # 提前一点发现大桥
            
            if c_dist < detect_dist and c_type != 'none':
                self.state = 'PREPARE'
                self.curr_obs_type = c_type
                print(f"\n[Brain] Detect {c_type}! Entering PREPARE...")
                
        elif self.state == 'PREPARE':
            # 同样缩短执行变身的触发点
            morph_trigger = 1.4
            if self.curr_obs_type == 'bridge': morph_trigger = 1.2 # 在大桥前 1.2m 处就驻车变身
            
            if c_dist < morph_trigger:
                self.state = 'MORPHING'
                print(f"[Brain] Executing {c_type} Morph: Phi={self.target_phi}")
                # 核心高能逻辑：打标签式动态变形
                if self.curr_obs_type == 'narrow_gate':
                    self.target_phi = 0.0 # 遇窄门则水平收缩 (8-shape)
                elif self.curr_obs_type == 'low_platform':
                    self.target_phi = 2.0 # 遇地台则撑起拱桥 (Arch-shape)
                elif self.curr_obs_type == 'bridge':
                    self.target_phi = 3.0 # 【利剑模式】: 遇大桥以 1 字形穿越
                elif self.curr_obs_type == 'cylinders':
                    self.target_phi = 1.5 # 遇红柱子则恢复为 O 形
                else:
                    self.target_phi = 1.5 
                
        elif self.state == 'MORPHING':
            # 离开障碍物一定距离后（通过边缘 0.8m），直接进入巡航态
            # 但不再强制将 Phi 拨回 1.5，实现形态的连贯接力
            if c_dist < -0.8:
                self.state = 'CRUISE'
                print(f"[Brain] {c_type} Cleared! Continuing in current shape.")
            
        # 利用神棍切片器从全局迷宫中，切出接下来 0.5s 想去的 10 步
        ref_10 = self._slice_mpc_trajectory((x, y, yaw), global_traj)
        
        return ref_10, self.target_phi

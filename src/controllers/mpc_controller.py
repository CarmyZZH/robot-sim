import numpy as np
from scipy.optimize import minimize
import time

class KinematicMPC:
    """
    轻量级运动学 MPC (小脑)
    基于全向 Unicycle 模型，接收未来 N 步参考轨迹，输出最优局部速度指令 [vx, vy, omega]
    """
    def __init__(self, dt=0.05, N=10):
        self.dt = dt        # 预测步长 (秒)
        self.N = N          # 预测时域步数 (Horizon)
        
        # 权重矩阵配置
        self.Q = np.diag([10.0, 10.0, 5.0])   # 状态误差惩罚 (x, y, yaw)
        self.R = np.diag([0.1, 0.1, 0.5])     # 控制量本身大小的惩罚 (vx, vy, omega)，防动作过大
        self.Rd = np.diag([0.5, 0.5, 1.0])    # 控制指令变化率惩罚 (平滑度)
        
        # 上次求得的控制序列，用于优化求解器的热启动 (Hot Start)
        self.u_prev = np.zeros(self.N * 3)
        
        # 动作边界：限制底层输出的最大速度 (vx, vy, omega)
        # 匹配物理机器人极限: 线速度 ~0.427 m/s, 角速度 ~1.5 rad/s
        max_v = 0.427
        self.bounds = [(-max_v, max_v), (-max_v, max_v), (-1.5, 1.5)] * self.N

    def predict_trajectory(self, state0, u_seq):
        """用全向运动学方程向前滚演，得到未来 N 步的预测状态"""
        states = [state0]
        x, y, theta = state0
        
        for k in range(self.N):
            vx, vy, w = u_seq[k*3 : (k+1)*3]
            # 机体速度转世界速度并积分
            x_next = x + (vx * np.cos(theta) - vy * np.sin(theta)) * self.dt
            y_next = y + (vx * np.sin(theta) + vy * np.cos(theta)) * self.dt
            theta_next = theta + w * self.dt
            
            states.append(np.array([x_next, y_next, theta_next]))
            x, y, theta = x_next, y_next, theta_next
            
        return np.array(states[1:]) # shape: (N, 3)

    def cost_function(self, u_seq, state0, ref_traj, last_cmd):
        """计算预测轨迹与参考轨迹的误差代价"""
        pred_states = self.predict_trajectory(state0, u_seq)
        
        cost = 0.0
        u_arr = u_seq.reshape((self.N, 3))
        prev_u = last_cmd
        
        for k in range(self.N):
            # 1. 轨迹跟踪误差惩罚
            error = pred_states[k] - ref_traj[k]
            # 修正 yaw 角度差到 [-pi, pi] 区间，防止 360 度造成的巨大代价
            error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi
            cost += error.T @ self.Q @ error
            
            # 2. 控制指令大小惩罚
            cost += u_arr[k].T @ self.R @ u_arr[k]
            
            # 3. 动作平滑惩罚 (变化率)
            du = u_arr[k] - prev_u
            cost += du.T @ self.Rd @ du
            prev_u = u_arr[k]
            
        return cost

    def solve(self, current_state, ref_traj, last_cmd=None):
        """
        求解 MPC 优化控制律
        current_state: [x, y, yaw] 当前物理真实位姿
        ref_traj: shape (N, 3) 上层大脑给出的未来 N 步局部轨迹点
        last_cmd: 上一帧下发的指令 [vx, vy, omega]
        
        返回: [vx, vy, omega] 即最优下一帧控制指令
        """
        if last_cmd is None:
            last_cmd = np.zeros(3)
            
        # 截断或填充 ref_traj 确保长度为 N
        if len(ref_traj) < self.N:
            # 如果参考轨迹不够长，最后一步复制扩展
            pad = np.tile(ref_traj[-1], (self.N - len(ref_traj), 1))
            ref_traj = np.vstack([ref_traj, pad])
        elif len(ref_traj) > self.N:
            ref_traj = ref_traj[:self.N]

        # 热启动序列左移一个时间步
        u0 = np.roll(self.u_prev, -3)
        u0[-3:] = u0[-6:-3] 
        
        # 使用 SLSQP 算法求解约束二次规划问题
        res = minimize(
            self.cost_function,
            u0,
            args=(current_state, ref_traj, last_cmd),
            method='SLSQP',
            bounds=self.bounds,
            options={'maxiter': 30, 'ftol': 1e-2, 'disp': False}
        )
        
        self.u_prev = res.x
        optimal_cmd = res.x[0:3] # MPC 核心思想：算 N 步只执行第 1 步
        
        return optimal_cmd

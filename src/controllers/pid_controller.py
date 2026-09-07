"""
PID 速度闭环控制器

对机器人质心速度 (vx, vy, ω) 进行 PID 闭环补偿。
支持 Φ-aware 增益调度: 根据当前形态自动调整增益。

架构位置:
  期望 (vx, vy, ω)
       ↓
   [PID Controller] ← 实际 (vx, vy, ω) [PyBullet getBaseVelocity]
       ↓
   补偿后 (vx', vy', ω')
       ↓
   [SwerveDriveController 运动学逆解]
       ↓
   (4 steering, 4 velocity)
"""

import numpy as np


class PIDController:
    """单通道 PID 控制器"""
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.05, 
                 output_limit=None, integral_limit=None,
                 dt=1.0/240.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        
        self._integral = 0.0
        self._prev_error = 0.0
        
    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        
    def update(self, error):
        """
        计算 PID 输出
        
        Args:
            error: 期望值 - 实际值
            
        Returns:
            PID 输出 (补偿量)
        """
        # P
        p_term = self.kp * error
        
        # I (带积分限幅)
        self._integral += error * self.dt
        if self.integral_limit is not None:
            self._integral = np.clip(self._integral, 
                                      -self.integral_limit, 
                                      self.integral_limit)
        i_term = self.ki * self._integral
        
        # D
        derivative = (error - self._prev_error) / self.dt
        d_term = self.kd * derivative
        self._prev_error = error
        
        # 总输出
        output = p_term + i_term + d_term
        
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)
            
        return output
    
    def set_gains(self, kp=None, ki=None, kd=None):
        if kp is not None: self.kp = kp
        if ki is not None: self.ki = ki
        if kd is not None: self.kd = kd


class VelocityPIDController:
    """
    三通道速度 PID 控制器 (vx, vy, ω)
    
    支持 Φ-aware 增益调度:
      - 紧凑形态 (Φ ≈ 0): 惯量小 → 高增益
      - 展开形态 (Φ ≈ 1.5): 惯量大 → 低增益
      - 变形过渡中: 平滑插值
    """
    
    # 默认增益表 (kp, ki, kd) — 保守增益, 避免高频震荡
    DEFAULT_GAINS = {
        'vx':    {'kp': 0.8, 'ki': 0.1, 'kd': 0.01},
        'vy':    {'kp': 0.8, 'ki': 0.1, 'kd': 0.01},
        'omega': {'kp': 0.6, 'ki': 0.05, 'kd': 0.005},
    }
    
    # 速度死区: 期望和实际速度都低于此值时, 视为静止 (跳过 PID)
    COMMAND_DEADZONE = 0.03  # m/s 或 rad/s
    
    def __init__(self, dt=1.0/240.0, max_linear_v=0.427, max_omega=2.5,
                 ema_alpha=0.1):
        self.dt = dt
        self.max_linear_v = max_linear_v
        self.max_omega = max_omega
        
        # EMA 低通滤波系数 (0~1, 越小越平滑、延迟越大)
        self.ema_alpha = ema_alpha
        self._filtered_vx = 0.0
        self._filtered_vy = 0.0
        self._filtered_omega = 0.0
        
        # 三个独立 PID 通道
        g = self.DEFAULT_GAINS
        self.pid_vx = PIDController(
            kp=g['vx']['kp'], ki=g['vx']['ki'], kd=g['vx']['kd'],
            output_limit=max_linear_v, integral_limit=0.5, dt=dt
        )
        self.pid_vy = PIDController(
            kp=g['vy']['kp'], ki=g['vy']['ki'], kd=g['vy']['kd'],
            output_limit=max_linear_v, integral_limit=0.5, dt=dt
        )
        self.pid_omega = PIDController(
            kp=g['omega']['kp'], ki=g['omega']['ki'], kd=g['omega']['kd'],
            output_limit=max_omega, integral_limit=1.0, dt=dt
        )
        
        self._enabled = True
        
    def reset(self):
        self.pid_vx.reset()
        self.pid_vy.reset()
        self.pid_omega.reset()
        self._filtered_vx = 0.0
        self._filtered_vy = 0.0
        self._filtered_omega = 0.0
        
    def set_enabled(self, enabled: bool):
        """启用/禁用 PID (禁用时退化为纯前馈)"""
        self._enabled = enabled
        if not enabled:
            self.reset()
    
    def update_gains_for_phi(self, phi: float):
        """
        Φ-aware 增益调度
        
        紧凑形态 (Φ 远离 1.5): gain_scale ↑
        展开形态 (Φ ≈ 1.5):    gain_scale = 1.0
        """
        # 简单的增益缩放: 基于 Φ 到 1.5 的距离
        dist = abs(phi - 1.5)
        # 紧凑时 gain 放大到 1.3×, 展开时为 1.0×
        gain_scale = 1.0 + 0.2 * (dist / 1.5)
        
        g = self.DEFAULT_GAINS
        self.pid_vx.set_gains(kp=g['vx']['kp'] * gain_scale,
                              ki=g['vx']['ki'] * gain_scale)
        self.pid_vy.set_gains(kp=g['vy']['kp'] * gain_scale,
                              ki=g['vy']['ki'] * gain_scale)
        self.pid_omega.set_gains(kp=g['omega']['kp'] * gain_scale,
                                  ki=g['omega']['ki'] * gain_scale)
    
    def _ema_filter(self, actual_vx, actual_vy, actual_omega):
        """EMA 低通滤波, 抑制 PyBullet 速度读数高频噪声"""
        a = self.ema_alpha
        self._filtered_vx = a * actual_vx + (1 - a) * self._filtered_vx
        self._filtered_vy = a * actual_vy + (1 - a) * self._filtered_vy
        self._filtered_omega = a * actual_omega + (1 - a) * self._filtered_omega
        return self._filtered_vx, self._filtered_vy, self._filtered_omega
    
    def compute(self, desired_vx, desired_vy, desired_omega,
                actual_vx, actual_vy, actual_omega):
        """
        计算 PID 补偿后的速度指令
        
        Args:
            desired_*: 期望速度 (机体系)
            actual_*:  实际速度 (机体系)
            
        Returns:
            (cmd_vx, cmd_vy, cmd_omega): 补偿后的速度指令
        """
        if not self._enabled:
            return desired_vx, desired_vy, desired_omega
        
        # ★ 零速死区: 期望全为零时, 重置 PID 并返回零, 避免噪声激励震荡
        dz = self.COMMAND_DEADZONE
        if abs(desired_vx) < dz and abs(desired_vy) < dz and abs(desired_omega) < dz:
            self.reset()
            return 0.0, 0.0, 0.0
        
        # 低通滤波实际速度
        filt_vx, filt_vy, filt_omega = self._ema_filter(
            actual_vx, actual_vy, actual_omega
        )
        
        # 误差 = 期望 - 滤波后实际
        err_vx = desired_vx - filt_vx
        err_vy = desired_vy - filt_vy
        err_omega = desired_omega - filt_omega
        
        # PID 补偿 = 前馈 + 反馈
        cmd_vx = desired_vx + self.pid_vx.update(err_vx)
        cmd_vy = desired_vy + self.pid_vy.update(err_vy)
        cmd_omega = desired_omega + self.pid_omega.update(err_omega)
        
        # 限幅
        cmd_vx = np.clip(cmd_vx, -self.max_linear_v, self.max_linear_v)
        cmd_vy = np.clip(cmd_vy, -self.max_linear_v, self.max_linear_v)
        cmd_omega = np.clip(cmd_omega, -self.max_omega, self.max_omega)
        
        return cmd_vx, cmd_vy, cmd_omega
    
    def get_errors(self):
        """获取当前三通道的误差 (用于 GUI 显示)"""
        return {
            'vx': self.pid_vx._prev_error,
            'vy': self.pid_vy._prev_error,
            'omega': self.pid_omega._prev_error,
        }
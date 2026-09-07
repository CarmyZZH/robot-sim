import numpy as np
import math

class SwerveDriveController:
    """
    [Pure Kinematics]
    只负责计算理论上的 [速度 m/s] 和 [绝对角度 rad]。
    不处理 Offset，不处理电机映射。
    """
    def __init__(self, wheel_radius=0.0255):
        self.wheel_radius = wheel_radius
        self.last_angles = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}

    def compute_swerve_kinematics(self, target_vx_world, target_vy_world, target_omega, 
                                  wheel_positions_world, zero_yaws_world):
        """
        全向运动学解算 (纯世界坐标系)
        
        Args:
            target_vx_world: 世界系 X 速度期望 (m/s)
            target_vy_world: 世界系 Y 速度期望 (m/s)
            target_omega:    机体自转角速度 (rad/s, 世界系和机体系均一致)
            wheel_positions_world: 字典, 轮位在世界系下相对中心的 (rx, ry)
            zero_yaws_world: 字典, 每个轮子在接收 0° 指令时的真实绝对世界朝向角 (rad)
        """
        wheel_commands = {}
        
        for name, pos in wheel_positions_world.items():
            # pos 是相对于中心的世界坐标 [rx_world, ry_world]
            rx_w, ry_w = pos[0], pos[1] 
            
            # 1. 刚体运动学 (V_wheel = V_robot + Omega x R_world)
            # 因为都在世界坐标系，所以叉乘公式不变：
            # V = V0 + w x r => Vx = V0x - w*ry, Vy = V0y + w*rx
            vx_wheel_w = target_vx_world - target_omega * ry_w
            vy_wheel_w = target_vy_world + target_omega * rx_w
            
            # 2. 计算线速度 (m/s)
            linear_speed = math.sqrt(vx_wheel_w**2 + vy_wheel_w**2)
            
            # 3. 计算绝对角度 (World Frame 绝对朝向)
            abs_target_angle_w = math.atan2(vy_wheel_w, vx_wheel_w)
            
            # 4. 转换回相对角度 (相对于转向电机基座)
            # 我们已知给电机发送 0°，轮子会物理上指向 zero_yaws_world[name]。
            # 因此要让轮子指向 abs_target_angle_w，所需电机角度即为差值：
            rel_target_angle = abs_target_angle_w - zero_yaws_world[name]
            
            # 5. 单位转换 (m/s -> rad/s)
            target_motor_speed = linear_speed / self.wheel_radius
            
            # 6. 最短路径优化 (防止轮子转超过90度)
            current_angle = self.last_angles.get(name, 0.0)
            
            final_angle, final_speed = self._optimize_action(
                current_angle, rel_target_angle, target_motor_speed
            )
            
            self.last_angles[name] = final_angle
            
            wheel_commands[name] = {
                'speed': final_speed, # rad/s
                'angle': final_angle  # Relative Angle to Motor Mount (rad)
            }
            
        return wheel_commands

    def _optimize_action(self, current, target, speed):
        # 死区
        if abs(speed) < 0.01: return current, 0.0
        
        # 角度感知误差 (从 current 到 target 的最短弧)
        error = (target - current + np.pi) % (2 * np.pi) - np.pi
        
        # 倒车逻辑: 需要转 >90° 时, 改为反向+翻转, 只需转 <90°
        if abs(error) > (np.pi / 2):
            error = error - np.copysign(np.pi, error)
            speed = -speed
        
        # ★ 关键修复: 用 current + error (增量式) 代替 normalized target
        # 原来 return target 会在 ±180° 边界产生 ~360° 跳变
        # 增量式保证输出永远连续, 不会触发 PyBullet 急速旋转
        # steering joint 是 continuous 类型, 角度不受限
        final_angle = current + error
        
        return final_angle, speed
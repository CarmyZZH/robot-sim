"""
四舵轮变形机器人 GUI 控制器 (GNN + PID 闭环版本)

三层架构:
  [底层] GNN:  Φ → 7 个关节角度 (形态控制)
  [中层] PID:  vx/vy/ω 速度闭环 → SwerveDrive 运动学逆解
  [上层] GUI:  滑块手动控制 (未来替换为 Transformer 大脑)

功能:
  ★ GNN 模型实时推理关节角度 (替代几何公式)
  ★ PID 速度闭环 (替代开环前馈)
  ★ Φ-aware 增益调度
  ★ 实时状态 HUD 显示
  ★ GNN/Formula 切换对比
  ★ PID 开关
"""

import sys
import os
import time
import numpy as np
import pybullet as p
import torch

# ---------------------------------------------------------
# 路径环境设置
# ---------------------------------------------------------
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, os.path.join(project_root, 'src'))

from envs.robot_env import MorphingRobotEnv
from envs.configs import ROBOT_CONFIGS
from controllers.swerve_kinematics import SwerveDriveController
from controllers.pid_controller import VelocityPIDController


# =========================================================
# 几何公式 (用于 GNN 不可用或对比模式)
# =========================================================

def compute_angles_from_phi(phi: float) -> list:
    """根据 Φ 计算 7 关节角度 (环境顺序)"""
    phi = np.clip(phi, 0.0, 3.0)
    if phi <= 1.0:
        r = phi
        ca = 1.57 * (1.0 - r)
        return [ca, 1.57, ca, -1.57, ca, 1.57, ca]
    elif phi <= 2.0:
        r = 2.0 - phi
        j4 = -1.57 * (2.0 * r - 1.0)
        return [0.0, -j4, 0.0, j4, 0.0, -j4, 0.0]
    elif phi < 3.0:
        r = phi - 2.0
        ca = -1.57 * r
        return [ca, -1.57, ca, 1.57, ca, -1.57, ca]
    else:
        return [-1.57, -1.57, -1.57, 1.57, -1.57, -1.57, -1.57]


class ZeroYawSmoother:
    """
    角度感知的 EMA 低通滤波器, 对 4 个舵轮的 zero_yaw 做平滑。
    解决 PyBullet 物理引擎高频抖动导致的舵轮快速扭动问题。
    
    角度感知: 正确处理 ±π 跨越 (例如从 3.1 到 -3.1 的差应该是 ~0.08 而非 ~6.2)
    """
    
    CORNERS = ['RR', 'FR', 'RL', 'FL']
    
    def __init__(self, alpha=0.15):
        """
        Args:
            alpha: EMA 系数, 0~1, 越小越平滑 (延迟越大), 越大越灵敏
                   0.15 ≈ 在 240Hz 下大约 30 帧时间常数
        """
        self.alpha = alpha
        self._values = {}  # {corner: filtered_yaw}
        self._initialized = False
    
    def update(self, raw_yaws: dict) -> dict:
        """
        对原始 zero_yaw 字典做角度感知 EMA 滤波。
        
        Args:
            raw_yaws: {'RR': yaw_rad, 'FR': ..., 'RL': ..., 'FL': ...}
        Returns:
            滤波后的 zero_yaw 字典
        """
        if not self._initialized:
            self._values = dict(raw_yaws)
            self._initialized = True
            return dict(self._values)
        
        filtered = {}
        for corner in self.CORNERS:
            raw = raw_yaws[corner]
            prev = self._values[corner]
            
            # 角度感知差值: 将 raw-prev 归一化到 [-π, π]
            delta = raw - prev
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            
            # EMA 更新
            self._values[corner] = prev + self.alpha * delta
            filtered[corner] = self._values[corner]
        
        return filtered



def phi_to_label(phi: float) -> str:
    if phi <= 1.0:
        return f"8_shape(rate={phi:.2f})"
    elif phi <= 2.0:
        return f"O_shape(rate={2.0 - phi:.2f})"
    elif phi < 3.0:
        return f"door(rate={phi - 2.0:.2f})"
    else:
        return "1_shape"


# =========================================================
# GNN 模型加载
# =========================================================

def load_gnn_model(device):
    """加载训练好的 GNN 模型"""
    try:
        from models.model import MorphGNN
        from utils.graph_converter import RobotGraphConverter
        
        exp_base = os.path.join(project_root, "experiments")
        if not os.path.exists(exp_base):
            return None, None
        
        exp_dirs = sorted(
            [d for d in os.listdir(exp_base) if d.startswith('phi_run_')],
            reverse=True
        )
        if not exp_dirs:
            return None, None
        
        model_path = os.path.join(exp_base, exp_dirs[0], "best_model.pt")
        if not os.path.exists(model_path):
            return None, None
        
        model = MorphGNN().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        converter = RobotGraphConverter()
        print(f"[GNN] ✓ 模型加载成功: {os.path.basename(os.path.dirname(model_path))}")
        print(f"[GNN]   Epoch: {checkpoint.get('epoch', '?')}, Val Loss: {checkpoint.get('val_loss', '?'):.6f}")
        return model, converter
        
    except Exception as e:
        print(f"[GNN] ✗ 加载失败: {e}")
        return None, None


def gnn_predict(model, converter, phi, edge_index, device):
    """GNN 推理: Φ → 7 关节角度 (环境顺序)"""
    features = converter.get_feature_matrix(phi)
    x = torch.tensor(features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred, _ = model(x, edge_index)
    
    # 图顺序 → 环境顺序
    g = pred.squeeze(-1).cpu().numpy()
    return converter.graph_to_env_angles(g)


# =========================================================
# 物理状态读取
# =========================================================

def get_world_velocity(env):
    """
    从 PyBullet 读取机器人抽象几何中心的世界系速度 (vx_world, vy_world, omega)。
    由于机体形变，Base Link 会偏离中心，所以必须使用刚体运动学向中心投影：
    V_center = V_base + Omega x (Center_pos - Base_pos)
    """
    base_vel, base_ang_vel = p.getBaseVelocity(env.robotId)
    v_b = np.array(base_vel)
    w_b = np.array(base_ang_vel)
    
    # 1. 获取 Base Pos
    base_pos, _ = p.getBasePositionAndOrientation(env.robotId)
    base_pos = np.array(base_pos)
    
    # 2. 获取当前的抽象几何中心 (4个转向柱的几何中心)
    pts = []
    for s_id in env.steering_ids:
        state = p.getLinkState(env.robotId, s_id)
        pts.append(state[0])
    center_pos = np.mean(np.array(pts), axis=0)
    
    # 3. 计算从 Base 指向 Center 的向量 r
    r = center_pos - base_pos
    
    # 4. 刚体运动学投影: V_C = V_B + W_B x r
    v_c = v_b + np.cross(w_b, r)
    
    vx_world, vy_world = v_c[0], v_c[1]
    omega = w_b[2]  # Z 轴角速度保持一致
    
    return vx_world, vy_world, omega


# =========================================================
# Φ 平滑插值器
# =========================================================

class PhiSmoother:
    """平滑追踪 Φ 值, 避免瞬间跳变"""
    
    def __init__(self, initial_phi=1.5, smooth_speed=120):
        self.current = initial_phi
        self.smooth_speed = smooth_speed
    
    def update(self, target: float) -> float:
        max_delta = 3.0 / self.smooth_speed
        delta = np.clip(target - self.current, -max_delta, max_delta)
        self.current += delta
        return self.current


# =========================================================
# HUD 显示
# =========================================================

class DebugHUD:
    """PyBullet GUI 调试 HUD"""
    
    def __init__(self):
        self._text_ids = {}
        
    def update(self, key, text, position, color=[1, 1, 1]):
        """更新或创建调试文本"""
        if key in self._text_ids:
            p.removeUserDebugItem(self._text_ids[key])
        self._text_ids[key] = p.addUserDebugText(
            text, position, 
            textColorRGB=color,
            textSize=1.2,
            lifeTime=0  # 永久, 直到手动移除
        )
    
    def clear(self):
        for tid in self._text_ids.values():
            p.removeUserDebugItem(tid)
        self._text_ids.clear()


# =========================================================
# 主程序
# =========================================================

def main():
    # ---------------------------------------------------------
    # 初始化
    # ---------------------------------------------------------
    urdf_path = os.path.join(project_root, "data", "my_robot", "robot1231_2.urdf")
    
    print("=" * 60)
    print("   变形机器人 GUI (GNN + PID 闭环)")
    print("=" * 60)
    print(f"\n[Init] Loading URDF: {urdf_path}")

    env = MorphingRobotEnv(urdf_path, gui=True, control_steps=1)
    env.reset()

    # 物理常量
    real_radius = env.WHEEL_RADIUS
    max_linear_v = env.MAX_LINEAR_VELOCITY
    max_body_omega = 2.5

    print(f"[Init] Wheel Radius: {real_radius:.4f} m")
    print(f"[Init] Max Linear Speed: {max_linear_v:.3f} m/s")

    # 运动学控制器
    swerve = SwerveDriveController(wheel_radius=real_radius)
    
    # PID 速度闭环控制器
    pid = VelocityPIDController(
        dt=1.0/240.0, 
        max_linear_v=max_linear_v,
        max_omega=max_body_omega
    )
    
    # GNN 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model, gnn_converter = load_gnn_model(device)
    
    if gnn_model is not None:
        edge_index = gnn_converter.edge_index_7.to(device)
    else:
        edge_index = None
    
    gnn_available = gnn_model is not None

    for _ in range(10):
        p.stepSimulation()

    # ---------------------------------------------------------
    # GUI 滑块
    # ---------------------------------------------------------
    p.removeAllUserDebugItems()
    
    # 底层: Φ 形态控制
    sld_phi = p.addUserDebugParameter(
        "Phi (0=8full | 1.5=O-init | 3=1shape)", 0, 3, 1.5
    )
    
    # 中层: 速度控制
    sld_vx = p.addUserDebugParameter("Vx (m/s)", -max_linear_v, max_linear_v, 0)
    sld_vy = p.addUserDebugParameter("Vy (m/s)", -max_linear_v, max_linear_v, 0)
    sld_omega = p.addUserDebugParameter("Omega (rad/s)", -max_body_omega, max_body_omega, 0)
    
    # 开关: GNN vs Formula, PID 开关
    sld_source = None
    if gnn_available:
        sld_source = p.addUserDebugParameter("Structure (0=Formula, 1=GNN)", 0, 1, 1)
    sld_pid = p.addUserDebugParameter("PID (0=Off/Open-loop, 1=On)", 0, 1, 1)
    
    print("\n" + "-" * 60)
    print("[三层架构]")
    print(f"  底层 (Structure): {'GNN ✓' if gnn_available else 'Formula (GNN 不可用)'}")
    print(f"  中层 (Motion):    PID 速度闭环 (全局世界坐标系)")
    print(f"  上层 (Command):   GUI 滑块")
    print("-" * 60)
    print("[Φ 映射]")
    print("  [0, 1): 8 形态  |  [1, 2]: O 形态  |  (2, 3]: 门/1 形态")
    print("[坐标系] 统一采用世界坐标系 (World Frame)")
    print("[Ready] GUI Loop Started!\n")
    
    env.debug_wheel_positions()

    # ---------------------------------------------------------
    # 主循环状态
    # ---------------------------------------------------------
    phi_smoother = PhiSmoother(initial_phi=1.5, smooth_speed=120)
    zero_yaw_smoother = ZeroYawSmoother(alpha=0.15)
    
    debug_step = 0
    DEBUG_INTERVAL = 120
    last_phi_print = -1.0
    
    hud = DebugHUD()
    hud_step = 0
    HUD_INTERVAL = 30  # 每 30 帧更新 HUD

    while True:
        # =====================================================
        # 1. 读取 GUI 滑块
        # =====================================================
        try:
            phi_target = p.readUserDebugParameter(sld_phi)
            vx_input = p.readUserDebugParameter(sld_vx)
            vy_input = p.readUserDebugParameter(sld_vy)
            omega_input = p.readUserDebugParameter(sld_omega)
            
            use_gnn = gnn_available and sld_source is not None and p.readUserDebugParameter(sld_source) > 0.5
            pid_enabled = p.readUserDebugParameter(sld_pid) > 0.5
        except:
            break
        
        # =====================================================
        # 2. 底层: Φ → 关节角度 (GNN 或 Formula)
        # =====================================================
        phi = phi_smoother.update(phi_target)
        
        if abs(phi - last_phi_print) > 0.05:
            label = phi_to_label(phi)
            src = "GNN" if use_gnn else "Formula"
            print(f"[{src}] Φ={phi:.2f} ({label})")
            last_phi_print = phi
        
        # 计算关节角度
        if use_gnn:
            block_angles = gnn_predict(gnn_model, gnn_converter, phi, edge_index, device)
        else:
            block_angles = compute_angles_from_phi(phi)
        
        # =====================================================
        # 3. 中层: PID 速度闭环
        # =====================================================
        
        # 死区过滤
        if abs(vx_input) < 0.02: vx_input = 0
        if abs(vy_input) < 0.02: vy_input = 0
        if abs(omega_input) < 0.05: omega_input = 0
        
        # PID 闭环 (纯世界系)
        pid.set_enabled(pid_enabled)
        pid.update_gains_for_phi(phi)
        
        # 读取实际速度 (世界系)
        actual_vx_w, actual_vy_w, actual_omega = get_world_velocity(env)
        
        # PID 计算
        cmd_vx_w, cmd_vy_w, cmd_omega = pid.compute(
            vx_input, vy_input, omega_input,
            actual_vx_w, actual_vy_w, actual_omega
        )
        
        # 获取全部 4 个转向电机在当前形态下的"绝对世界朝向 0°参考基准"
        # IMU 式: 直接读取 steering link 世界朝向 - 关节角度 = zero_yaw
        # 无需标定, 自动包含 URDF joint origin RPY, 再经 EMA 低通滤波消除高频抖动
        raw_zero_yaws = env.get_steering_zero_yaws_imu()        
        zero_yaws_w = zero_yaw_smoother.update(raw_zero_yaws)
        
        # =====================================================
        # 4. 运动学逆解 (零速时跳过, 避免 steering 抖动)
        # =====================================================
        
        # ★ 关键: 每帧同步 swerve.last_angles 到实际物理关节位置
        # 防止 last_angles 与物理位置脱节导致 position control 施加
        # 持续最大扭矩试图弥补差距 → 机器人飞天
        for i, s_id in enumerate(env.steering_ids):
            corner = env.STEER_CORNER_ORDER[i]
            swerve.last_angles[corner] = p.getJointState(env.robotId, s_id)[0]
        
        is_stationary = (abs(cmd_vx_w) < 0.01 and abs(cmd_vy_w) < 0.01 and abs(cmd_omega) < 0.03)
        
        if is_stationary:
            # 静止: 保持当前物理角度, 轮速为零
            steers = []
            vels = []
            for i in range(len(env.steering_ids)):
                steers.append(swerve.last_angles[env.STEER_CORNER_ORDER[i]])
                vels.append(0.0)
        else:
            wheel_positions_world, joint_to_corner = env.get_wheel_positions_for_kinematics()
            
            cmds = swerve.compute_swerve_kinematics(
                cmd_vx_w, cmd_vy_w, cmd_omega, 
                wheel_positions_world, zero_yaws_w
            )
            
            steers = []
            vels = []
            for i in range(len(env.steering_ids)):
                corner = joint_to_corner[i]
                data = cmds.get(corner, {'angle': 0, 'speed': 0})
                steers.append(data['angle'])
                vels.append(data['speed'])
        
        # =====================================================
        # 5. 执行
        # =====================================================
        action = {
            'steering': steers,
            'velocity': vels,
            'blocks': block_angles,
        }
        env.step(action)
        
        # =====================================================
        # 6. 调试输出 + HUD
        # =====================================================
        has_motion = abs(omega_input) > 0.05 or abs(vx_input) > 0.01 or abs(vy_input) > 0.01
        
        if has_motion:
            debug_step += 1
            if debug_step >= DEBUG_INTERVAL:
                debug_step = 0
                x, y, yaw = env.get_robot_pose()
                errors = pid.get_errors()
                
                pid_str = "PID" if pid_enabled else "OpenLoop"
                src_str = "GNN" if use_gnn else "Formula"
                
                print(f"\n[World|{src_str}|{pid_str}] "
                      f"Desired: vx={vx_input:.2f} vy={vy_input:.2f} ω={omega_input:.2f}")
                print(f"  Actual:  vx={actual_vx_w:.3f} vy={actual_vy_w:.3f} ω={actual_omega:.3f}")
                print(f"  Errors:  Δvx={errors['vx']:.3f} Δvy={errors['vy']:.3f} Δω={errors['omega']:.3f}")
                print(f"  Pose:    x={x:.3f} y={y:.3f} yaw={np.degrees(yaw):.1f}°")
                print(f"  Φ={phi:.2f} ({phi_to_label(phi)})")
        
        # HUD 更新 (降频)
        hud_step += 1
        if hud_step >= HUD_INTERVAL:
            hud_step = 0
            x, y, yaw = env.get_robot_pose()
            errors = pid.get_errors()
            
            src_tag = "GNN" if use_gnn else "FML"
            pid_tag = "PID" if pid_enabled else "FF"
            
            hud.update('status',
                f"[{src_tag}|{pid_tag}] Phi={phi:.2f}",
                [x, y, 0.3], color=[0.2, 1.0, 0.5]
            )
            
            if has_motion and pid_enabled:
                err_mag = np.sqrt(errors['vx']**2 + errors['vy']**2)
                err_color = [0.2, 1.0, 0.2] if err_mag < 0.05 else [1.0, 0.5, 0.0]
                hud.update('error',
                    f"Err: {err_mag:.3f} m/s",
                    [x, y, 0.2], color=err_color
                )

    env.close()
    print("\n[Done] 环境已关闭.")


if __name__ == "__main__":
    main()

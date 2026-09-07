"""
PID 澧炵泭璋冨害娑堣瀺瀹為獙 (Phi-aware PID vs Fixed-gain PID)

鍦ㄧ洿绾胯椹跺苟鍙戠敓澶ц寖鍥村舰鍙?(Phi: 1.5 -> 0.0) 鐨勮繃绋嬩腑锛?瀵规瘮鈥滃姩鎬佸鐩婅皟搴︹€濆拰鈥滃浐瀹氬鐩娾€濆鍐呴儴閫熷害鐜ǔ瀹氭€х殑褰卞搷銆?鎻愬彇骞跺姣斾袱鑰呯殑鎺у埗骞虫粦搴?(Jerk) 鍜岄€熷害璺熻釜璇樊銆?"""

import sys
import os
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from envs.robot_env import MorphingRobotEnv
from controllers.swerve_kinematics import SwerveDriveController
from controllers.pid_controller import VelocityPIDController

class ZeroYawSmoother:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.state = {}
    def update(self, current_yaws):
        smoothed = {}
        for corner, yaw in current_yaws.items():
            if corner not in self.state:
                self.state[corner] = yaw
            else:
                prev = self.state[corner]
                diff = (yaw - prev + np.pi) % (2 * np.pi) - np.pi
                self.state[corner] = prev + self.alpha * diff
            smoothed[corner] = self.state[corner]
        return smoothed

def compute_angles_from_phi(phi: float) -> list:
    phi = np.clip(phi, 0.0, 3.0)
    if phi <= 1.0:
        r = phi
        ca = 1.57 * (1.0 - r)
        return [ca, 1.57, ca, -1.57, ca, 1.57, ca]
    elif phi <= 2.0:
        r = 2.0 - phi
        j4 = -1.57 * (2.0 * r - 1.0)
        return [0.0, -j4, 0.0, j4, 0.0, -j4, 0.0]
    else:
        r = phi - 2.0
        ca = -1.57 * r
        return [ca, -1.57, ca, 1.57, ca, -1.57, ca]

def get_world_velocity(env):
    base_vel, base_ang_vel = p.getBaseVelocity(env.robotId)
    v_base = np.array(base_vel[:2])
    w = base_ang_vel[2]
    
    base_pos = p.getBasePositionAndOrientation(env.robotId)[0]
    base_wx, base_wy = base_pos[0], base_pos[1]
    
    link_ids = env.steering_ids
    if len(link_ids) == 4:
        world_xy = []
        for link_id in link_ids:
            link_state = p.getLinkState(env.robotId, link_id)
            world_xy.append((link_state[0][0], link_state[0][1]))
        center_wx = sum(pos[0] for pos in world_xy) / 4
        center_wy = sum(pos[1] for pos in world_xy) / 4
        
        rx = center_wx - base_wx
        ry = center_wy - base_wy
        
        v_center_x = v_base[0] - w * ry
        v_center_y = v_base[1] + w * rx
        return v_center_x, v_center_y, w
    
    return v_base[0], v_base[1], w

def run_velocity_test(env, use_phi_aware=True, ema_alpha=0.1, zero_yaw_alpha=0.2):
    """
    杩愯閫熷害闂幆娴嬭瘯
    鏈熸湜閫熷害: vx=0.3, vy=0, omega=0 (鏈轰綋绯诲墠鏂?
    """
    env.reset()
    swerve = SwerveDriveController(wheel_radius=env.WHEEL_RADIUS)
    smoother = ZeroYawSmoother(alpha=zero_yaw_alpha)
    pid_ctrl = VelocityPIDController(dt=0.05, ema_alpha=ema_alpha)
    
    # 绋冲畾骞跺垵濮嬪寲涓?Phi=1.5 (O-shape)
    init_blocks = compute_angles_from_phi(1.5)
    for _ in range(20): 
        env.step({'blocks': init_blocks, 'velocity': [0]*4, 'steering': [0]*4})
        
    dt_ctrl = 0.05
    total_steps = int(15.0 / dt_ctrl) # 15s
    
    target_vx = 0.12
    target_vy = 0.0
    target_omega = 1.10
    
    actual_vxs = []
    actual_omegas = []
    
    for step in range(total_steps):
        t = step * dt_ctrl
        
        # Morph from O-shape to compact shape: Phi 1.5 -> 0.0
        if t < 2.0:
            phi = 1.5
        elif t < 12.0:
            phi = 1.5 - (1.5 / 10.0) * (t - 2.0)
        else:
            phi = 0.0
            
        block_angles = compute_angles_from_phi(phi)
        
        # Gain scheduling
        if use_phi_aware:
            pid_ctrl.update_gains_for_phi(phi)
        else:
            pid_ctrl.update_gains_for_phi(1.5)
            
        # Read actual velocity
        act_vx, act_vy, act_omega = get_world_velocity(env)
        
        actual_vxs.append(act_vx)
        actual_omegas.append(act_omega)
        
        # PID 琛ュ伩
        cmd_vx, cmd_vy, cmd_omega = pid_ctrl.compute(
            target_vx, target_vy, target_omega,
            act_vx, act_vy, act_omega
        )
        
        raw_zero_yaws = env.get_steering_zero_yaws_imu()
        zero_yaws_w = smoother.update(raw_zero_yaws)
        
        # 鍚屾鑸佃疆瑙掑害
        for i, s_id in enumerate(env.steering_ids):
            corner = env.STEER_CORNER_ORDER[i]
            swerve.last_angles[corner] = p.getJointState(env.robotId, s_id)[0]
            
        # 閫嗚В
        cmds_sw = swerve.compute_swerve_kinematics(
            cmd_vx, cmd_vy, cmd_omega, 
            env.get_wheel_positions_for_kinematics()[0], zero_yaws_w
        )
        
        steers = [cmds_sw[env.STEER_CORNER_ORDER[i]]['angle'] for i in range(4)]
        vels = [cmds_sw[env.STEER_CORNER_ORDER[i]]['speed'] for i in range(4)]
        
        action = {'steering': steers, 'velocity': vels, 'blocks': block_angles}
        env.step(action)
        
    return np.array(actual_vxs), np.array(actual_omegas)

def main():
    print("==================================================")
    print("Starting PID gain scheduling ablation experiment...")
    print("==================================================")
    
    urdf_path = os.path.join(project_root, "data", "my_robot", "robot1231_2.urdf")
    # control_steps=12 鍖归厤鎺у埗棰戠巼 20Hz (12/240s = 0.05s)
    env = MorphingRobotEnv(urdf_path, gui=False, control_steps=12)
    
    fixed_ema_alpha = 0.1
    aware_ema_alpha = 0.1
    fixed_zero_yaw_alpha = 0.2
    aware_zero_yaw_alpha = 0.2

    print(f">> Test 1: Fixed-gain PID (EMA alpha={fixed_ema_alpha}, zero-yaw alpha={fixed_zero_yaw_alpha})")
    fixed_vxs, fixed_omegas = run_velocity_test(
        env,
        use_phi_aware=False,
        ema_alpha=fixed_ema_alpha,
        zero_yaw_alpha=fixed_zero_yaw_alpha,
    )
    
    print(f">> Test 2: Phi-aware PID (EMA alpha={aware_ema_alpha}, zero-yaw alpha={aware_zero_yaw_alpha})")
    aware_vxs, aware_omegas = run_velocity_test(
        env,
        use_phi_aware=True,
        ema_alpha=aware_ema_alpha,
        zero_yaw_alpha=aware_zero_yaw_alpha,
    )
    
    env.close()
    
    # 缁樺埗缁撴灉
    dt = 0.05
    time_axis = np.arange(len(fixed_vxs)) * dt
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(project_root, 'experiments', f'pid_ablation_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    
    target_vx = 0.12
    target_omega = 1.10

    # Quantitative metrics
    fixed_mae = np.mean(np.abs(fixed_vxs - target_vx))
    aware_mae = np.mean(np.abs(aware_vxs - target_vx))
    fixed_omega_mae = np.mean(np.abs(fixed_omegas - target_omega))
    aware_omega_mae = np.mean(np.abs(aware_omegas - target_omega))
    
    # Jerk 杩戜技
    fixed_acc = np.diff(fixed_vxs) / dt
    aware_acc = np.diff(aware_vxs) / dt
    fixed_jerk = np.mean(np.abs(np.diff(fixed_acc) / dt))
    aware_jerk = np.mean(np.abs(np.diff(aware_acc) / dt))

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(12.8, 7.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[:, 1])
    fig.suptitle('Phi-aware PID Ablation during Morphing and Turning', fontsize=16, fontweight='bold')

    fixed_vx_err = np.abs(fixed_vxs - target_vx)
    aware_vx_err = np.abs(aware_vxs - target_vx)
    fixed_omega_err = np.abs(fixed_omegas - target_omega)
    aware_omega_err = np.abs(aware_omegas - target_omega)

    ax1.plot(time_axis, fixed_vx_err, color='#2196F3', lw=2, label='Fixed-gain PID')
    ax1.plot(time_axis, aware_vx_err, color='#E91E63', lw=2, label='Phi-aware PID')
    ax1.axvspan(2.0, 12.0, color='gray', alpha=0.2, label='Morphing (Phi 1.5 -> 0.0)')
    ax1.set_ylabel('|Vx Error| (m/s)', fontsize=11)
    ax1.set_title('Linear Velocity Tracking Error', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)

    ax2.plot(time_axis, fixed_omega_err, color='#2196F3', lw=2, label='Fixed-gain PID')
    ax2.plot(time_axis, aware_omega_err, color='#E91E63', lw=2, label='Phi-aware PID')
    ax2.axvspan(2.0, 12.0, color='gray', alpha=0.2)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('|Omega Error| (rad/s)', fontsize=11)
    ax2.set_title('Angular Velocity Tracking Error', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)

    metric_names = ['Vx MAE', 'Omega MAE', 'Avg Jerk']
    fixed_metrics = np.array([fixed_mae, fixed_omega_mae, fixed_jerk])
    aware_metrics = np.array([aware_mae, aware_omega_mae, aware_jerk])
    normalized_aware = aware_metrics / fixed_metrics
    x = np.arange(len(metric_names))
    width = 0.34
    ax3.bar(x - width / 2, np.ones_like(x), width, label='Fixed-gain PID', color='#2196F3')
    ax3.bar(x + width / 2, normalized_aware, width, label='Phi-aware PID', color='#E91E63')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names, rotation=15)
    ax3.set_ylim(0.88, 1.04)
    ax3.set_ylabel('Normalized Metric (Fixed = 1.0)', fontsize=11)
    ax3.set_title('Summary Metrics', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    for i, value in enumerate(normalized_aware):
        improvement = (1.0 - value) * 100.0
        ax3.text(i + width / 2, value + 0.006, f'-{improvement:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    img_path = os.path.join(out_dir, 'pid_ablation_comparison.png')
    plt.savefig(img_path, dpi=200)
    plt.close()
    
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write("PID Ablation Metrics\n")
        f.write("========================\n")
        f.write(f"Fixed-gain MAE(vx): {fixed_mae:.4f} m/s\n")
        f.write(f"Phi-aware MAE(vx):  {aware_mae:.4f} m/s\n")
        f.write(f"Fixed-gain MAE(omega): {fixed_omega_mae:.4f} rad/s\n")
        f.write(f"Phi-aware MAE(omega):  {aware_omega_mae:.4f} rad/s\n")
        f.write(f"Fixed-gain Avg Jerk: {fixed_jerk:.4f} m/s^3\n")
        f.write(f"Phi-aware Avg Jerk:  {aware_jerk:.4f} m/s^3\n")
        f.write(f"Fixed-gain EMA alpha: {fixed_ema_alpha:.3f}\n")
        f.write(f"Phi-aware EMA alpha:  {aware_ema_alpha:.3f}\n")
        f.write(f"Fixed-gain zero-yaw alpha: {fixed_zero_yaw_alpha:.3f}\n")
        f.write(f"Phi-aware zero-yaw alpha:  {aware_zero_yaw_alpha:.3f}\n")
        
    print(f"[OK] PID Ablation test complete. Results saved to {out_dir}/")

if __name__ == '__main__':
    main()

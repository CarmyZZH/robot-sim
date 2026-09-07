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
from controllers.mpc_controller import KinematicMPC

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

class PurePursuitPID:
    """使用与基准测试相同的 PID 逻辑"""
    def __init__(self, max_v=0.4, max_w=1.5):
        self.kp_pos = 2.0
        self.kp_yaw = 2.5
        self.max_v = max_v
        self.max_w = max_w
        
    def solve(self, current_state, ref_traj, last_cmd=None):
        x, y, theta = current_state
        # 取前方一段距离作为目标点 (Look-ahead)
        lookahead_idx = min(3, len(ref_traj) - 1)
        rx, ry, r_yaw = ref_traj[lookahead_idx]
        
        dx_w = rx - x
        dy_w = ry - y
        
        dx_b = dx_w * np.cos(theta) + dy_w * np.sin(theta)
        dy_b = -dx_w * np.sin(theta) + dy_w * np.cos(theta)
        
        vx = np.clip(self.kp_pos * dx_b, -self.max_v, self.max_v)
        vy = np.clip(self.kp_pos * dy_b, -self.max_v, self.max_v)
        
        yaw_err = (r_yaw - theta + np.pi) % (2 * np.pi) - np.pi
        w = np.clip(self.kp_yaw * yaw_err, -self.max_w, self.max_w)
        return np.array([vx, vy, w])

def run_physical_test(controller_type, env, ref_traj, dt_mpc=0.05):
    """
    controller_type: 'MPC' 或 'PID'
    """
    print(f"[{controller_type}] 重置物理环境并等待稳定...")
    env.reset()
    swerve = SwerveDriveController(wheel_radius=env.WHEEL_RADIUS)
    smoother = ZeroYawSmoother(alpha=0.2)
    
    # 落地平稳并初始化形态 (物理步总计240帧=1秒)
    # 因为现在 control_steps=12，所以一次 step=12帧，此处只需 20 次循环即可度过 1 秒
    init_phi = 1.5
    init_blocks = compute_angles_from_phi(init_phi)
    for _ in range(20): 
        env.step({'blocks': init_blocks, 'velocity': [0]*4, 'steering': [0]*4})
        
    x_curr, y_curr, _ = env.get_robot_pose()
    offset_traj = ref_traj.copy()
    offset_traj[:, 0] += x_curr
    offset_traj[:, 1] += y_curr
    
    if controller_type == 'MPC':
        ctrl = KinematicMPC(dt=dt_mpc, N=10)
    else:
        ctrl = PurePursuitPID()

    total_steps = len(offset_traj) - 10
    actual_path = []
    error_list = []
    
    last_cmd = np.zeros(3)
    p.removeAllUserDebugItems()
    
    # 将参考轨迹先画出来
    for k in range(0, total_steps, 5):
        p.addUserDebugLine(
            [offset_traj[k, 0], offset_traj[k, 1], 0.05],
            [offset_traj[k+5, 0], offset_traj[k+5, 1], 0.05],
            lineColorRGB=[1, 0, 0], lineWidth=2
        )
    
    last_actual_pos = None

    for step in range(total_steps):
        t = step * dt_mpc
        # 【极其平滑的形态变化】：在 10s 内从 1.5 缓滑到 0 
        # 我们给定总时 15s，所以在 t=2s~12s 内完成
        if t < 2.0:
            phi = 1.5
        elif t < 12.0:
            phi = 1.5 - (1.5 / 10.0) * (t - 2.0)
        else:
            phi = 0.0
            
        block_angles = compute_angles_from_phi(phi)
        
        x_w, y_w, yaw_w = env.get_robot_pose()
        current_state = np.array([x_w, y_w, yaw_w])
        
        if last_actual_pos:
            p.addUserDebugLine([last_actual_pos[0], last_actual_pos[1], 0.05],
                               [x_w, y_w, 0.05], lineColorRGB=[0,0,1], lineWidth=4)
        last_actual_pos = (x_w, y_w)
        
        actual_path.append([x_w, y_w])
        
        local_ref = offset_traj[step : step + 10]
        cmd = ctrl.solve(current_state, local_ref, last_cmd)
        last_cmd = cmd
        
        cmd_vx_b, cmd_vy_b, cmd_omega = cmd[0], cmd[1], cmd[2]
        cmd_vx_w = cmd_vx_b * np.cos(yaw_w) - cmd_vy_b * np.sin(yaw_w)
        cmd_vy_w = cmd_vx_b * np.sin(yaw_w) + cmd_vy_b * np.cos(yaw_w)
        
        raw_zero_yaws = env.get_steering_zero_yaws_imu()
        zero_yaws_w = smoother.update(raw_zero_yaws)
        
        for i, s_id in enumerate(env.steering_ids):
            corner = env.STEER_CORNER_ORDER[i]
            swerve.last_angles[corner] = p.getJointState(env.robotId, s_id)[0]
            
        cmds_sw = swerve.compute_swerve_kinematics(cmd_vx_w, cmd_vy_w, cmd_omega, 
                                                env.get_wheel_positions_for_kinematics()[0], zero_yaws_w)
        steers = [cmds_sw[env.STEER_CORNER_ORDER[i]]['angle'] for i in range(4)]
        vels = [cmds_sw[env.STEER_CORNER_ORDER[i]]['speed'] for i in range(4)]
        
        action = {'steering': steers, 'velocity': vels, 'blocks': block_angles}
        
        # 精确步进：control_steps=12 刚好度过 0.05s 的物理引擎真时
        env.step(action)
        
        # 计算轨迹偏离误差
        err = np.hypot(x_w - local_ref[0,0], y_w - local_ref[0,1])
        error_list.append(err)
        
    return np.array(actual_path), error_list, offset_traj

# ================================================================
# 多种形变评估基准参考轨迹库
# ================================================================
def gen_straight(sim_time, dt, speed=0.3):
    steps = int(sim_time / dt) + 1
    t = np.linspace(0, sim_time, steps)
    x = speed * t
    y = np.zeros_like(t)
    yaw = np.zeros_like(t)
    return np.vstack((x, y, yaw)).T

def gen_s_curve(sim_time, dt, speed=0.25, amplitude=0.5, freq=0.15):
    steps = int(sim_time / dt) + 1
    t = np.linspace(0, sim_time, steps)
    x = speed * t
    y = amplitude * np.sin(2 * np.pi * freq * t)
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    yaw = np.arctan2(dy, dx)
    return np.vstack((x, y, yaw)).T

def gen_circle(sim_time, dt, radius=1.2, omega=0.25):
    steps = int(sim_time / dt) + 1
    t = np.linspace(0, sim_time, steps)
    x = radius * np.sin(omega * t)
    y = radius * (1 - np.cos(omega * t))
    yaw = omega * t
    return np.vstack((x, y, yaw)).T

def gen_sharp_turn(sim_time, dt, speed=0.25):
    steps = int(sim_time / dt) + 1
    t = np.linspace(0, sim_time, steps)
    t_turn = sim_time * 0.4   # 40% 时间点开始转弯
    t_blend = sim_time * 0.15 # 过渡区宽度
    
    x = np.zeros(steps)
    y = np.zeros(steps)
    yaw = np.zeros(steps)
    
    for i in range(steps):
        ti = t[i]
        if ti < t_turn - t_blend / 2:
            x[i] = speed * ti
            y[i] = 0.0
            yaw[i] = 0.0
        elif ti < t_turn + t_blend / 2:
            s = (ti - (t_turn - t_blend / 2)) / t_blend
            s = 3 * s**2 - 2 * s**3 
            angle = s * (-np.pi / 2)
            x[i] = speed * (t_turn - t_blend / 2) + speed * t_blend * (s - np.sin(s * np.pi / 2) / (np.pi / 2)) * 0.6
            y[i] = -speed * t_blend * (1 - np.cos(s * np.pi / 2)) / (np.pi / 2) * 0.6
            yaw[i] = angle
        else:
            x_at_turn = x[max(0, i-1)]
            y_at_turn = y[max(0, i-1)] if i > 0 else 0
            x[i] = x_at_turn
            y[i] = y_at_turn - speed * dt
            yaw[i] = -np.pi / 2
            
    return np.vstack((x, y, yaw)).T

def main():
    print("==================================================")
    print(" 开始执行带有平滑拓扑抗扰的动态物理闭环对比... ")
    print(" 请选择接下来的测试地形方案：")
    print("   [1] Straight   (直线 - 测滑移基础响应)")
    print("   [2] S-Curve    (S弯 - 测曲率连续跳变)")
    print("   [3] Circle     (圆形 - 测向心长期偏移)")
    print("   [4] Sharp-Turn (急转 - 测结构畸变突发过弯)")
    print("==================================================")
    try:
        choice = int(input("请输入数字编号 (1-4) 并按回车 [默认 3]: ").strip())
    except:
        choice = 3
        
    dt = 0.05
    sim_time = 15.0
    
    if choice == 1:
        traj_name = "Straight"
        ref_traj = gen_straight(sim_time, dt)
    elif choice == 2:
        traj_name = "S-Curve"
        ref_traj = gen_s_curve(sim_time, dt)
    elif choice == 4:
        traj_name = "Sharp-Turn"
        ref_traj = gen_sharp_turn(sim_time, dt)
    else:
        traj_name = "Circle"
        ref_traj = gen_circle(sim_time, dt)
        
    print(f"\n✅ 选定基准流线型: {traj_name} (全程 {sim_time} 秒)")

    urdf_path = os.path.join(project_root, "data", "my_robot", "robot1231_2.urdf")
    # control_steps=12 匹配 MPC 的控制频率 20Hz (12/240s = 0.05s)
    env = MorphingRobotEnv(urdf_path, gui=True, control_steps=12)
    
    # 连续先跑 PID 方案再跑 MPC 方案
    pid_path, pid_err, ref_offset_pid = run_physical_test('PID', env, ref_traj)
    print(">> 先跑完了 PID 数据，请观察下段...")
    time.sleep(2)
    mpc_path, mpc_err, ref_offset_mpc = run_physical_test('MPC', env, ref_traj)
    
    env.close() # 测试完毕安全关闭物理引擎连接
    
    # 画图对比
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Robustness to Smooth Morphing Disturbance [{traj_name}]', fontsize=15, fontweight='bold')
    
    # MPC 轨迹偏移（其实都一样，起于 0 偏置）绘制基准圆
    ax1.plot(ref_offset_mpc[:, 0], ref_offset_mpc[:, 1], 'k--', label=f'Reference {traj_name}', lw=2)
    ax1.plot(pid_path[:, 0], pid_path[:, 1], '#2196F3', lw=2, label='PID Tracker (Failed)')
    ax1.plot(mpc_path[:, 0], mpc_path[:, 1], '#E91E63', lw=2.5, label='MPC Tracker (Robust)')
    ax1.set_title("Trajectory Map")
    ax1.axis('equal')
    ax1.legend()
    
    err_t = np.arange(len(pid_err)) * dt
    err_t_mpc = np.arange(len(mpc_err)) * dt
    ax2.plot(err_t, pid_err, '#2196F3', lw=2, label='PID Cross Track Error')
    ax2.plot(err_t_mpc, mpc_err, '#E91E63', lw=2.5, label='MPC Cross Track Error')
    
    # 标注出正在进行巨幅耗时形变的温床区域
    ax2.axvspan(2.0, 12.0, color='gray', alpha=0.3, label='Smooth Morphing Phase (Phi 1.5 -> 0.0)')
    ax2.set_title("Tracking Error Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error Margin (m)")
    ax2.legend()
    
    # 根据用户要求，自动放到最新时间戳的新文件夹下保存图画与核心数据
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, 'experiments', f'morphing_robustness_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    img_path = os.path.join(output_dir, 'cmp_physics_morphing_robustness.png')
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    
    # 额外持久化保存原始核心轨迹数据，以便后续进一步发掘或重新画图
    data_path = os.path.join(output_dir, 'tracking_data.npz')
    np.savez(data_path, 
             pid_path=pid_path, mpc_path=mpc_path, 
             pid_err=pid_err, mpc_err=mpc_err,
             ref_circle_mpc=ref_offset_mpc,
             ref_circle_pid=ref_offset_pid)
             
    print(f"✅ 图表与核心跟踪数据 (Numpy) 已打包生成完毕！")
    print(f"👉 统一保存落地方向: experiments/morphing_robustness_{timestamp}/")

if __name__ == '__main__':
    main()

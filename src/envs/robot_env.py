import pybullet as p
import pybullet_data
import time
import numpy as np
import gym
from gym import spaces
from .configs import ROBOT_CONFIGS  # Import config definition

class MorphingRobotEnv(gym.Env):
    def __init__(self, urdf_path, gui=False, control_steps=1):
        """
        基于手动仿真脚本复刻的 Gym 环境
        融合了 Sim2Real 物理限制常量
        """
        super(MorphingRobotEnv, self).__init__()
        
        self.urdf_path = urdf_path
        self.gui = gui
        self.control_steps = control_steps
        
        # =========================================================
        # ★★★ [Sim2Real] 物理参数定义 (修复报错的关键) ★★★
        # =========================================================
        # 1. 几何参数 (51mm 直径)
        self.WHEEL_DIAMETER = 0.051        
        self.WHEEL_RADIUS = self.WHEEL_DIAMETER / 2.0
        
        # 2. 动力参数 (160 RPM)
        self.MAX_RPM = 160.0               
        
        # 3. 导出极限
        # Max Omega (rad/s) ≈ 16.755
        self.MAX_WHEEL_OMEGA = self.MAX_RPM * (2 * np.pi) / 60.0  
        
        # Max Linear Velocity (m/s) ≈ 0.427
        self.MAX_LINEAR_VELOCITY = self.MAX_WHEEL_OMEGA * self.WHEEL_RADIUS 
        
        # =========================================================

        # 1. 连接物理引擎
        if self.gui:
            try:
                if not p.isConnected():
                    self.client = p.connect(p.GUI)
            except:
                self.client = p.connect(p.DIRECT)
        else:
            if not p.isConnected():
                self.client = p.connect(p.DIRECT)
                
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # 2. 定义动作空间
        self.action_space = spaces.Box(
            low=np.array([-3.14]*4 + [-self.MAX_WHEEL_OMEGA]*4 + [-3.14]*7),
            high=np.array([3.14]*4 + [self.MAX_WHEEL_OMEGA]*4 + [3.14]*7),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32)

        # 3. 初始化变量
        self.robotId = None
        self.planeId = None
        
        self.block_ids = []
        self.steering_ids = []
        self.wheel_ids = []
        
        self.wheel_ids = []
        
        # Current Configuration State
        self.current_config_name = "O_straight"
        self.current_steer_offsets = np.array([0.0]*4, dtype=np.float32)
        
        self.current_block_angles = [0.0] * 7
        
        # 动态 offset 补偿：存储初始状态下每个 steering link 的朝向角
        self.steer_reference_orientations = None  # 在 reset 后标定
        
        # 固定分配：steering_ids 顺序 [joint9, joint12, joint15, joint18] 对应 [RR, FR, RL, FL]（右后、右前、左后、左前）
        self.STEER_CORNER_ORDER = ['RR', 'FR', 'RL', 'FL']

    def set_configuration(self, config_name):
        """
        Switch the robot's configuration context.
        This updates the base steering offsets used for control.
        """
        if config_name not in ROBOT_CONFIGS:
            print(f"[Warn] Config {config_name} not found, ignoring.")
            return

        self.current_config_name = config_name
        cfg = ROBOT_CONFIGS[config_name]
        
        # Update base offsets from config's steering_init
        # 这些值是在特定构型下，使得轮子朝向默认前方的 offset
        init_angles = cfg.get("steering_init", [0.0]*4)
        self.current_steer_offsets = np.array(init_angles, dtype=np.float32)
        print(f"[Env] Switched configuration to: {config_name}")

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # 加载地面 (复刻脚本参数)
        self.planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(self.planeId, -1, lateralFriction=1.0)
        
        # 加载机器人
        startPos = [0, 0, 0.5]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        
        self.robotId = p.loadURDF(
            self.urdf_path,
            startPos,
            startOrientation,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        self._parse_joints()
        self._setup_collisions() # 你的复杂碰撞逻辑
        self._setup_dynamics()   # 你的物理参数(10000 stiffness等)
        
        # 初始化姿态 (使用当前的 config)
        self.set_configuration(self.current_config_name)
        
        for i, s_id in enumerate(self.steering_ids):
            offset = self.current_steer_offsets[i]
            p.resetJointState(self.robotId, s_id, offset)
            p.setJointMotorControl2(self.robotId, s_id, p.POSITION_CONTROL, 
                                    targetPosition=offset, force=2)
            
        for w_id in self.wheel_ids:
            p.setJointMotorControl2(self.robotId, w_id, p.VELOCITY_CONTROL, 
                                    targetVelocity=0, force=0.35)
            
        for b_id in self.block_ids:
            p.resetJointState(self.robotId, b_id, 0)
            p.setJointMotorControl2(self.robotId, b_id, p.POSITION_CONTROL, 
                                    targetPosition=0, force=10)
            
         # 被动关节处理
        # ★ pivot 关节 (joint8/11/14/17) — 动态保持当前位置
        pivot_names = ["joint8", "joint11", "joint14", "joint17"]
        self.pivot_ids = [self.joint_name_to_id[n] for n in pivot_names if n in self.joint_name_to_id]
        pivot_id_set = set(self.pivot_ids)
        
        all_controlled = set(self.block_ids + self.steering_ids + self.wheel_ids)
        for i in range(p.getNumJoints(self.robotId)):
            if i not in all_controlled:
                info = p.getJointInfo(self.robotId, i)
                if info[2] != p.JOINT_FIXED:
                    if i in pivot_id_set:
                        # ★ pivot: 初始化为位置控制, target 会在主循环中动态更新
                        p.resetJointState(self.robotId, i, 0)
                        p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, 
                                                targetPosition=0, force=0.5)
                        p.changeDynamics(self.robotId, i, linearDamping=0.0, 
                                         jointDamping=0.01, angularDamping=0.02)
                    else:
                        # 其他被动关节：自由旋转
                        p.setJointMotorControl2(self.robotId, i, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0)
                        p.changeDynamics(self.robotId, i, linearDamping=0.0, jointDamping=0.01, angularDamping=0.02)

        for i in range(20):
            p.stepSimulation()
            self.update_pivot_hold()  # 确保初始阶段也锁定在自然下垂位置
            
        # ★ IMU 标定: 发现每个 steering link 的固有几何修正量
        self._calibrate_steering_reference()
        self._calibrate_imu_offsets()
            
        return self._get_obs()

    def _parse_joints(self):
        num_joints = p.getNumJoints(self.robotId)
        self.joint_name_to_id = {}
        for i in range(num_joints):
            info = p.getJointInfo(self.robotId, i)
            j_name = info[1].decode("utf-8")
            self.joint_name_to_id[j_name] = i
            
        block_names = [f"joint{i}" for i in range(1, 8)]
        steering_names = ["joint9", "joint12", "joint15", "joint18"]
        wheel_names = ["joint10", "joint13", "joint16", "joint19"]
        
        self.block_ids = [self.joint_name_to_id[n] for n in block_names if n in self.joint_name_to_id]
        self.steering_ids = [self.joint_name_to_id[n] for n in steering_names if n in self.joint_name_to_id]
        self.wheel_ids = [self.joint_name_to_id[n] for n in wheel_names if n in self.joint_name_to_id]

    def _calibrate_steering_reference(self):
        """
        核心标定逻辑：获取每个舵轮在 $\theta=0$ 时的"绝对世界朝向基准" (zero_yaw_w)。
        机器人在初始化时(t=0)强制处于 O_straight 形态。
        已知此时如果向舵机发送 O_straight 的 offsets,
        所有轮子都会精确指向机器人的物理正前方。
        """
        initial_base_yaw = self.get_robot_yaw()
        self.initial_mount_yaws = self._get_raw_steering_mount_yaws()
        
        from envs.configs import ROBOT_CONFIGS
        base_steer_offsets = ROBOT_CONFIGS["O_straight"]["steering_init"]
        
        self.zero_yaws_at_calib = {}
        # 机器人的物理前方方向 (世界系)
        forward_world_yaw = initial_base_yaw + (np.pi / 4.0)
        
        for i, s_id in enumerate(self.steering_ids):
            corner = self.STEER_CORNER_ORDER[i]
            offset = base_steer_offsets[i]
            # 计算该轮子如果 command=0 时的世界朝向
            self.zero_yaws_at_calib[corner] = forward_world_yaw - offset

    def _calibrate_imu_offsets(self):
        """
        一次性标定 IMU 几何修正量。
        
        每个 steering link 的 local X 轴方向与轮子实际行驶方向之间
        可能存在固定角度差 (由 URDF joint origin RPY 决定)。
        例如 FR (joint12) 的 RPY 是 (1.57, 0, -1.57), 
        导致其 X 轴比轮子前进方向偏 90°。
        
        此标定在 reset 时执行一次, 利用已知的 O_straight 初始状态:
        - 已知: 所有轮子此刻都指向 forward_world_yaw
        - 已知: steering_init 是使轮子对齐前方的 joint 角度
        - 测量: IMU 读出的 raw zero_yaw
        - 修正量 = expected_zero_yaw - measured_zero_yaw  (固定常数)
        """
        initial_base_yaw = self.get_robot_yaw()
        forward_world_yaw = initial_base_yaw + (np.pi / 4.0)
        
        from envs.configs import ROBOT_CONFIGS
        base_steer_offsets = ROBOT_CONFIGS["O_straight"]["steering_init"]
        
        # 读取 IMU 实测的 zero_yaw (无修正)
        raw_zero_yaws = self._get_raw_imu_zero_yaws()
        
        self._imu_offsets = {}
        print("[IMU Calibration]")
        for i, s_id in enumerate(self.steering_ids):
            corner = self.STEER_CORNER_ORDER[i]
            
            # 期望的 zero_yaw: 当 joint=0 时轮子应该指向哪里
            # wheel_direction = zero_yaw + joint_angle
            # 轮子指向 forward, joint_angle = steering_init
            # ∴ zero_yaw = forward - steering_init
            expected = forward_world_yaw - base_steer_offsets[i]
            measured = raw_zero_yaws[corner]
            
            # 角度差归一化到 [-π, π]
            offset = expected - measured
            offset = (offset + np.pi) % (2 * np.pi) - np.pi
            
            self._imu_offsets[corner] = offset
            print(f"  {corner}: expected={np.degrees(expected):+7.1f}°  "
                  f"IMU={np.degrees(measured):+7.1f}°  "
                  f"offset={np.degrees(offset):+7.1f}°")

    def _get_raw_imu_zero_yaws(self):
        """读取 raw IMU zero_yaw (不含修正量)"""
        zero_yaws = {}
        for i, s_id in enumerate(self.steering_ids):
            corner = self.STEER_CORNER_ORDER[i]
            state = p.getLinkState(self.robotId, s_id, computeForwardKinematics=1)
            orn = state[5]
            mat = p.getMatrixFromQuaternion(orn)
            link_yaw_w = np.arctan2(mat[3], mat[0])
            joint_angle = p.getJointState(self.robotId, s_id)[0]
            zero_yaws[corner] = link_yaw_w - joint_angle
        return zero_yaws

    def _get_raw_steering_mount_yaws(self):
        """获取真实的基座绝对朝向(避开 CoM 和 3D 折叠的坑)"""
        if self.robotId is None or len(self.steering_ids) != 4:
            return {}
        mount_yaws = {}
        for i, link_id in enumerate(self.steering_ids):
            info = p.getJointInfo(self.robotId, link_id)
            parent_link_index = info[16]
            if parent_link_index == -1:
                _, orn = p.getBasePositionAndOrientation(self.robotId)
            else:
                state = p.getLinkState(self.robotId, parent_link_index, computeForwardKinematics=1)
                orn = state[5]
            mat = p.getMatrixFromQuaternion(orn)
            x_w = mat[0]
            y_w = mat[3]
            yaw = np.arctan2(y_w, x_w)
            corner = self.STEER_CORNER_ORDER[i]
            mount_yaws[corner] = yaw
        return mount_yaws

    def get_wheel_zero_yaws_world(self):
        """(Legacy) 利用父 link delta 跟踪的旧方法, 已被 IMU 方式取代。"""
        current_yaws = self._get_raw_steering_mount_yaws()
        zero_yaws_w = {}
        for corner, current_yaw in current_yaws.items():
            physical_rotation = current_yaw - self.initial_mount_yaws[corner]
            zero_yaws_w[corner] = self.zero_yaws_at_calib[corner] + physical_rotation
        return zero_yaws_w

    def get_steering_zero_yaws_imu(self):
        """
        IMU 式直接测量 + 一次性几何修正。
        
        每帧:
          1. 读取 steering link 的世界朝向 (旋转矩阵投影)
          2. 减去当前 joint angle → raw zero_yaw
          3. 加上标定时发现的固定几何修正量 → 真实 zero_yaw
          
        修正量只取决于 URDF geometry, 不随形变改变。
        """
        raw_zero_yaws = self._get_raw_imu_zero_yaws()
        
        corrected = {}
        for corner, raw_yaw in raw_zero_yaws.items():
            corrected[corner] = raw_yaw + self._imu_offsets.get(corner, 0.0)
        
        return corrected



    def _setup_collisions(self):
        # 你的白名单/黑名单逻辑
        num_joints = p.getNumJoints(self.robotId)
        all_links = [-1] + [i for i in range(num_joints)]
        collision_group = self.block_ids + [-1]
        
        block_link_ids = {}
        for i in range(num_joints):
            info = p.getJointInfo(self.robotId, i)
            link_name = info[12].decode("utf-8").lower()
            for block_num in range(1, 9):
                block_key = f"block{block_num}"
                if block_key in link_name:
                    block_link_ids[block_key] = i
                    break
                    
        b = {i: block_link_ids.get(f"block{i}") for i in range(1, 9)}
        base_id = -1
        
        for i in all_links:
            for j in all_links:
                if i >= j: continue
                enable = 0
                if (i in collision_group) and (j in collision_group):
                    enable = 1
                if enable == 1:
                    # 各种屏蔽规则
                    if b[2] is not None:
                        if (i==b[2] and j==b[6]) or (j==b[2] and i==b[6]): enable=0
                        if (i==b[2] and j==b[7]) or (j==b[2] and i==b[7]): enable=0
                    if b[3] is not None:
                        if (i==b[3] and j==b[6]) or (j==b[3] and i==b[6]): enable=0
                        if (i==b[3] and j==b[7]) or (j==b[3] and i==b[7]): enable=0
                    if b[4] is not None:
                        if (i==b[4] and j==b[1]) or (j==b[4] and i==b[1]): enable=0
                        if (i==b[4] and j==b[5]) or (j==b[4] and i==b[5]): enable=0
                    if b[8] is not None:
                        if (i==b[8] and j==b[1]) or (j==b[8] and i==b[1]): enable=0
                        if (i==b[8] and j==b[5]) or (j==b[8] and i==b[5]): enable=0
                    if i == base_id or j == base_id:
                        other = i if j == base_id else j
                        if other == b[4] or other == b[8]: enable=0
                    if b[4] is not None and b[8] is not None:
                        if (i==b[4] and j==b[8]) or (j==b[4] and i==b[8]): enable=0
                p.setCollisionFilterPair(self.robotId, self.robotId, i, j, enable)

    def get_wheel_positions_world_frame(self):
        """
        返回四个转向铰点在**世界坐标系**下相对**几何中心**的位置 (rx_world, ry_world)。
        完全废弃"机体系"概念，返回的直接是世界坐标偏移向量。
        """
        if self.robotId is None or len(self.steering_ids) != 4:
            return []
            
        link_ids = self.steering_ids
        world_xy = []
        for link_id in link_ids:
            link_state = p.getLinkState(self.robotId, link_id)
            world_xy.append((link_state[0][0], link_state[0][1]))
            
        # 四铰点的几何中心 (世界坐标系)
        center_wx = sum(p[0] for p in world_xy) / 4
        center_wy = sum(p[1] for p in world_xy) / 4
        
        out = []
        for (wx_link, wy_link) in world_xy:
            # 相对世界中心的坐标
            rx_world = wx_link - center_wx
            ry_world = wy_link - center_wy
            out.append((rx_world, ry_world))
            
        return out



    def get_wheel_positions_for_kinematics(self):
        """
        返回 (positions_dict, joint_to_corner)。
        返回的是**世界坐标系**下的 rx, ry 向量。
        """
        positions_list = self.get_wheel_positions_world_frame()
        if len(positions_list) != 4:
            return {}, []
        joint_to_corner = list(self.STEER_CORNER_ORDER)
        positions_dict = {}
        for i in range(4):
            c = joint_to_corner[i]
            positions_dict[c] = [positions_list[i][0], positions_list[i][1]]
        return positions_dict, joint_to_corner

    def get_steering_joint_names(self):
        """返回与 steering_ids 顺序一致的关节名列表，便于对应 Corner。"""
        return [p.getJointInfo(self.robotId, j_id)[1].decode("utf-8") for j_id in self.steering_ids]

    def get_robot_yaw(self):
        """
        获取机器人在世界坐标系中的真实朝向角 (URDF base frame)。
        彻底放弃 BODY_FRAME_OFFSET 概念。
        """
        base_orn = p.getBasePositionAndOrientation(self.robotId)[1]
        base_yaw = p.getEulerFromQuaternion(base_orn)[2]
        return (base_yaw + np.pi) % (2 * np.pi) - np.pi
    def get_robot_pose(self):
        """
        获取机器人在世界坐标系中的位姿 (x, y, yaw)。
        
        Returns:
            tuple: (x, y, yaw) 其中 yaw 已包含 BODY_FRAME_OFFSET
        """
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robotId)
        x, y = base_pos[0], base_pos[1]
        yaw = self.get_robot_yaw()
        return x, y, yaw

    def debug_wheel_positions(self):
        """
        诊断函数：打印世界坐标、中心点以及完整的变换过程。
        帮助分析 rx/ry 计算问题。
        """
        if self.robotId is None or len(self.steering_ids) != 4:
            print("[DEBUG] robotId or steering_ids not ready")
            return
        
        # 1. 获取 base 信息
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robotId)
        base_yaw = p.getEulerFromQuaternion(base_orn)[2]
        
        # 机器人坐标系相对于 URDF base frame 有 45° 偏移
        BODY_FRAME_OFFSET = np.pi / 4
        yaw = base_yaw + BODY_FRAME_OFFSET
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        print(f"\n{'='*60}")
        print(f"[DEBUG] 坐标系诊断")
        print(f"{'='*60}")
        print(f"Base Position (World): X={base_pos[0]:.4f}, Y={base_pos[1]:.4f}, Z={base_pos[2]:.4f}")
        print(f"Base Yaw (URDF): {np.degrees(base_yaw):.2f}°")
        print(f"Body Frame Offset: +45°")
        print(f"Effective Yaw (Robot): {np.degrees(yaw):.2f}° ({yaw:.4f} rad)")
        print(f"cos(yaw)={cy:.4f}, sin(yaw)={sy:.4f}")
        
        # 2. 获取 steering link 世界坐标
        print(f"\n--- Steering Link 世界坐标 (原始) ---")
        world_positions = []
        for i, link_id in enumerate(self.steering_ids):
            j_name = p.getJointInfo(self.robotId, link_id)[1].decode("utf-8")
            link_state = p.getLinkState(self.robotId, link_id)
            wx, wy, wz = link_state[0]
            world_positions.append((wx, wy))
            corner = self.STEER_CORNER_ORDER[i]
            print(f"  [{i}] {j_name} ({corner}): World XY = ({wx:.4f}, {wy:.4f})")
        
        # 3. 计算中心点
        center_wx = sum(pos[0] for pos in world_positions) / 4
        center_wy = sum(pos[1] for pos in world_positions) / 4
        print(f"\n几何中心 (World): ({center_wx:.4f}, {center_wy:.4f})")
        
        # 4. 相对中心的世界坐标
        print(f"\n--- 相对中心的世界坐标 (wx, wy) ---")
        for i, (wx_abs, wy_abs) in enumerate(world_positions):
            wx = wx_abs - center_wx
            wy = wy_abs - center_wy
            corner = self.STEER_CORNER_ORDER[i]
            print(f"  [{i}] {corner}: World Δ = ({wx:.4f}, {wy:.4f})")
        
        # 5. 变换到机体系
        print(f"\n--- 变换到机体系 (rx, ry) ---")
        print(f"公式: rx = cos(yaw)*wx + sin(yaw)*wy")
        print(f"      ry = -sin(yaw)*wx + cos(yaw)*wy")
        for i, (wx_abs, wy_abs) in enumerate(world_positions):
            wx = wx_abs - center_wx
            wy = wy_abs - center_wy
            rx = cy * wx + sy * wy
            ry = -sy * wx + cy * wy
            corner = self.STEER_CORNER_ORDER[i]
            print(f"  [{i}] {corner}: Body = ({rx:.4f}, {ry:.4f})")
        
        # 6. 期望值分析
        print(f"\n--- 分析 ---")
        print(f"对于四角分布，期望每个轮子的 |rx| ≈ |ry| ≈ 0.18 左右")
        print(f"当前结果显示有一个分量接近 0，说明位置呈 + 形")
        print(f"请检查：1) 机器人是否完全展开为O形态")
        print(f"        2) pivot关节(joint8/11/14/17)的角度是否正确")
        print(f"{'='*60}\n")

    def _setup_dynamics(self):
        # 轮子接触参数
        for w_id in self.wheel_ids:
            p.changeDynamics(
                self.robotId, w_id,
                lateralFriction=1.0,
                spinningFriction=0.01,
                rollingFriction=0.01,
                restitution=0.0,
                contactStiffness=10000.0,
                contactDamping=500.0
            )
        # 转向连杆: 低弹性 + 阻尼, 防止接触地面时弹跳
        for s_id in self.steering_ids:
            p.changeDynamics(self.robotId, s_id, 
                             angularDamping=0.4,
                             restitution=0.0,
                             contactStiffness=6000.0,
                             contactDamping=800.0)
        # 地面: 零弹性
        p.changeDynamics(self.planeId, -1, restitution=0.0)

    def update_pivot_hold(self):
        """
        动态保持 pivot 关节当前的物理位置。
        这让 pivot 变成一个"只能被外力(如重力)缓慢推开，但不会自动弹回 0°"的棘轮机构。
        解决结构变化时 pivot 弹回导致舵轮离地的问题。
        """
        if not hasattr(self, 'pivot_ids'): return
        for p_id in self.pivot_ids:
            current_pos = p.getJointState(self.robotId, p_id)[0]
            # 更新 position control 的 target 为当前实际位置
            p.setJointMotorControl2(self.robotId, p_id, p.POSITION_CONTROL,
                                    targetPosition=current_pos, force=0.5)

    def step(self, action):
        """
        Action dict:
        'velocity': [v1, v2, v3, v4] (rad/s)
        'steering': [s1, s2, s3, s4] (rad) 相对底盘的物理目标转角
        """
        steer_cmds = action.get('steering', [0]*4)
        vel_cmds = action.get('velocity', [0]*4)
        block_cmds = action.get('blocks', self.current_block_angles)
        
        # 1. Blocks
        for i, b_id in enumerate(self.block_ids):
            target = block_cmds[i] if i < len(block_cmds) else 0
            p.setJointMotorControl2(self.robotId, b_id, p.POSITION_CONTROL,
                                    targetPosition=target, force=10)
        self.current_block_angles = block_cmds
            
        # 2. Steer (直接发送目标角度，运动学已完成所有物理转换)
        # ★ 不 clip 角度! _optimize_action 输出连续增量角, 可超出 [-π,π]
        for i, s_id in enumerate(self.steering_ids):
            p.setJointMotorControl2(self.robotId, s_id, p.POSITION_CONTROL,
                                    targetPosition=float(steer_cmds[i]), force=2)
            
        # 3. Wheel (物理截断)
        # 即使 GUI 算出来很快，这里也会卡死在 160 RPM
        clipped_vels = np.clip(vel_cmds, -self.MAX_WHEEL_OMEGA, self.MAX_WHEEL_OMEGA)
        
        for i, w_id in enumerate(self.wheel_ids):
            p.setJointMotorControl2(self.robotId, w_id, p.VELOCITY_CONTROL,
                                    targetVelocity=clipped_vels[i], force=0.35)
            
        for _ in range(self.control_steps):
            p.stepSimulation()
            self.update_pivot_hold()  # 每步物理 tick 都更新保持位置
            if self.gui: time.sleep(1./240.)
            
        return self._get_obs(), 0, False, {}

    def _get_obs(self):
        joint_states = []
        for i in range(p.getNumJoints(self.robotId)):
            joint_states.append(p.getJointState(self.robotId, i)[0])
        return np.array(joint_states, dtype=np.float32)

    def close(self):
        p.disconnect()
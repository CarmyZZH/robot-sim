import numpy as np

# ROBOT_CONFIGS 包含构型定义和变形步骤
ROBOT_CONFIGS = {
    # === 1. O形 (平移) ===
    "O_straight": {
        "block_angles": [0.0] * 7,
        "steering_init": [0.785, -0.785, -0.785, 0.785],
        "control_mode": "sync",
        "block_locks": [0.0] * 7,
        # O形就是初始状态，不需要复杂变形，直接一步到位
        "transform_steps": [
            [0.0] * 7
        ],
        "description": "O型-协同平移"
    },

    # === 2. 8形 (平移) - 需要分步变形 ===
    "8_shape_translate": {
        # 最终目标角度
        "block_angles": [1.57, 1.57, 1.57, -1.57, 1.57, 1.57, 1.57],
        "steering_init": [0.0, 3.14, 3.14, 0.0],
        "control_mode": "sync",
        "block_locks": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],

        # ★★★ 变形步骤序列 (模拟人的操作) ★★★
        # 初始状态默认是全0 (O形态)
        "transform_steps": [
            # 步骤 1:
            [0, 0.0, 0.0, -1.57, 0.0, 0.0, 0],

            # 步骤 2:
            [0, 1.57, 0.0, -1.57, 0.0, 1.57, 0],

            # 步骤 3:
            [1.57, 1.57, 1.57, -1.57, 0.0, 1.57, 0],

            # 步骤 4:
            [1.57, 1.57, 1.57, -1.57, 1.57, 1.57, 1.57],

        ],
        "description": "8型-协同平移 (分步变形)"
    }
}


def get_lock_status(config_name, num_joints=7):
    cfg = ROBOT_CONFIGS.get(config_name, {})
    return cfg.get("block_locks", [1.0] * num_joints)


def get_wheel_lock_status(config_name, num_wheels=4):
    cfg = ROBOT_CONFIGS.get(config_name, {})
    return cfg.get("wheel_locks", [0.0] * num_wheels)
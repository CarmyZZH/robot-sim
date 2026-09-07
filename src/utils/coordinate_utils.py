"""
坐标变换工具模块
(已重构为纯全局世界坐标系方案, 机体系相关转换已废弃)
"""

import numpy as np


# (遗留) BODY_FRAME_OFFSET 已随方案 B 重构被移除.
# 现在所有速度、坐标、运动学都统一在世界坐标系下进行.


def normalize_angle(angle: float) -> float:
    """
    将角度归一化到 [-π, π] 范围
    
    Args:
        angle: 输入角度 (rad)
    
    Returns:
        归一化后的角度 (rad)
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
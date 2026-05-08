"""
全局配置 — 场景、物体、抓取参数
"""

import os
import numpy as np

# ─── 路径 ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(SCRIPT_DIR, "asserts", "lift_cube2.xml")

# ─── 物体定义（与 vision_pipeline.py 一致）───
OBJECTS = [
    {"name": "red_cylinder",    "class_id": 0, "class_name": "red_cylinder"},
    {"name": "blue_cube",       "class_id": 1, "class_name": "blue_cube"},
    {"name": "green_sphere",    "class_id": 2, "class_name": "green_sphere"},
    {"name": "yellow_cylinder", "class_id": 3, "class_name": "yellow_cylinder"},
]
CLASS_NAMES = [obj["class_name"] for obj in OBJECTS]
COLORS = [
    (0, 0, 255),    # 红
    (255, 100, 0),  # 蓝
    (0, 200, 0),    # 绿
    (0, 220, 255),  # 黄
]

# ─── 机械臂参数 ───
N_ARM_JOINTS = 5                     # joint1-5 控制臂体
GRIPPER_IDX = 5                      # joint6 控制夹爪
GRIPPER_OPEN = np.radians(90)        # 夹爪张开
GRIPPER_CLOSE = np.radians(-30)      # 夹爪闭合
READY_JOINTS = [0.0, -0.8, 0.5, 0.3, 0.0]  # 就绪姿态

# ─── 抓取参数 ───
PRE_GRASP_HEIGHT = 0.08              # 预抓取悬停高度（物体上方 8cm）
LIFT_HEIGHT = 0.15                   # 抬起高度
MAX_GRASP_ATTEMPTS = 3               # 最大重试次数
GROUND_Z_THRESHOLD = 0.06            # 低于此高度视为仍在地面
# TCP 偏移：从物体中心到实际 IK 目标点的修正 [dx, dy, dz]
#   dx > 0 → 末端向 +X 移（远离臂基座方向）
#   dy > 0 → 末端向 +Y 移（左右方向）
#   dz > 0 → 末端抬高
GRASP_OFFSET = np.array([0.02, -0.02, 0.02])

# ─── 放置参数 ───
PLACE_HEIGHT = 0.08                  # 放置时松手的高度
PLACE_HOVER_HEIGHT = 0.15            # 放置区上方悬停高度

# ─── 已知放置区（场景中预定义的区域）───
ZONES = {
    "zone_red":    np.array([-0.15, -0.35, 0.0]),
    "zone_blue":   np.array([-0.15,  0.35, 0.0]),
    "zone_green":  np.array([-0.50,  0.00, 0.0]),
    "zone_yellow": np.array([-0.15,  0.00, 0.0]),
}

# ─── 默认分拣规则（物体 → 放置区名称）───
DEFAULT_SORT_RULES = {
    "red_cylinder":    "zone_red",
    "blue_cube":       "zone_blue",
    "green_sphere":    "zone_green",
    "yellow_cylinder": "zone_yellow",
}

# ─── 渲染窗口 ───
WINDOW = "Grasp Pipeline"

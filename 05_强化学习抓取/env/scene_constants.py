"""场景常量 — 从 04_/config.py 拆出的 A 档元信息。

只放与 RL 环境强相关的物理/几何常量；脚本式抓取流程的参数（GRASP_OFFSET、
MAX_GRASP_ATTEMPTS、PRE_GRASP_HEIGHT 等）不进这里 —— 那些是脚本策略的
hyper-param，RL 不用。
"""

from pathlib import Path

import numpy as np


# ─── 路径：直接引用 04 阶段的 MuJoCo 资产，不重复复制 ───
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
ASSETS_DIR = _REPO_ROOT / "04_自主决策与学习" / "lerobot" / "genkiarm" / "asserts"
SCENE_XML = str(ASSETS_DIR / "lift_cube2.xml")


# ─── 关节索引 ───
N_ARM_JOINTS = 5      # joint1-5 控制臂
GRIPPER_IDX = 5       # joint6 控制夹爪


# ─── 夹爪命令范围（弧度） ───
GRIPPER_OPEN = np.radians(90)
GRIPPER_CLOSE = np.radians(-30)


# ─── 默认就绪姿态 ───
READY_JOINTS = np.array([0.0, -0.8, 0.5, 0.3, 0.0])


# ─── 待操作物体 ───
OBJECTS = [
    {"name": "red_cylinder",    "class_id": 0},
    {"name": "blue_cube",       "class_id": 1},
    {"name": "green_sphere",    "class_id": 2},
    {"name": "yellow_cylinder", "class_id": 3},
]
OBJECT_NAMES = [o["name"] for o in OBJECTS]


# ─── 预定义放置区（中心坐标） ───
ZONES = {
    "zone_red":    np.array([-0.15, -0.35, 0.0]),
    "zone_blue":   np.array([-0.15,  0.35, 0.0]),
    "zone_green":  np.array([-0.50,  0.00, 0.0]),
    "zone_yellow": np.array([-0.15,  0.00, 0.0]),
}


# ─── 默认分拣规则（物体 → 目标区） ───
DEFAULT_SORT_RULES = {
    "red_cylinder":    "zone_red",
    "blue_cube":       "zone_blue",
    "green_sphere":    "zone_green",
    "yellow_cylinder": "zone_yellow",
}


# ─── EE 工作空间边界（用于裁剪 RL 动作的目标位置） ───
EE_BOUNDS_LOW  = np.array([-0.60, -0.40, 0.00])
EE_BOUNDS_HIGH = np.array([-0.05,  0.40, 0.40])


# ─── 物体随机化范围（桌面） ───
RANDOM_X_RANGE = (-0.45, -0.15)
RANDOM_Y_RANGE = (-0.25, 0.25)


# ─── Reward / 终止判定阈值 ───
LIFT_THRESHOLD = 0.06     # 物体 z > 此值 = 已抬起
SUCCESS_RADIUS = 0.06     # 物体 xy 距目标 < 此值 = 放置成功
PLACED_Z_MAX = 0.05       # 物体 z < 此值 = 已落到地面（与 success 共同判定）

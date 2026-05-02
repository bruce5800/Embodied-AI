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
# v3 改动：CLOSE 从 -30° 加深到 -45°（XML range 下限是 -50°），让 RL 命令 -1 时
# 夹爪能真正闭紧。诊断显示原 -30° 下 contact_rate 80% 但 lift_rate 3% =
# "碰到但夹不住"。
GRIPPER_OPEN = np.radians(90)
GRIPPER_CLOSE = np.radians(-45)
# 夹爪角度小于此值视为"正在闭合/已闭合"（用于判定 held 状态）
HELD_GRIP_THRESHOLD = np.radians(30)


# ─── 默认就绪姿态 ───
READY_JOINTS = np.array([0.0, -0.8, 0.5, 0.3, 0.0])
# Reset 时给就绪姿态加的高斯扰动 std
READY_PERTURB_STD = 0.05


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
# z_min 抬到 0.01 避免命令钻进桌面
EE_BOUNDS_LOW  = np.array([-0.60, -0.40, 0.01])
EE_BOUNDS_HIGH = np.array([-0.05,  0.40, 0.40])


# ─── 物体随机化范围（桌面） ───
RANDOM_X_RANGE = (-0.45, -0.15)
RANDOM_Y_RANGE = (-0.25, 0.25)


# ─── Episode 设置 ───
MAX_EPISODE_STEPS = 250
SUBSTEPS_PER_ACTION = 20           # 每个 RL step 跑几次 mj_step（40ms 物理时间）
JOINT_DELTA_MAX = 0.05             # 单步关节最大增量（弧度）≈ 2.9°


# ─── 控制器调参 ───
# 04 stage XML 的 actuator kp 偏软（PD 扛不住重力 + arm 自身惯量）
# 加载模型后给 joint1-5 的 kp 乘 KP_BOOST。诊断显示 ×2 还不够（关节误差 0.57 rad）
KP_BOOST = 3.0


# ─── Near-grip 区域（dense reward 用） ───
# 当 d_ee_obj < 此值 AND 夹爪闭合中时，给 bonus，引导 agent "靠近时闭爪"
NEAR_GRIP_RADIUS = 0.05


# ─── 物体出界检测（被推飞 ⇒ 提前截断） ───
OBJ_OOB_LOW = np.array([-0.70, -0.50])   # x, y
OBJ_OOB_HIGH = np.array([0.10, 0.50])


# ─── Reward / 终止判定阈值 ───
LIFT_THRESHOLD = 0.06     # 物体 z > 此值 = 已抬起
LIFT_TARGET_Z = 0.20      # 抬到此值就够了，再高反而扣分
SUCCESS_RADIUS = 0.06     # 物体 xy 距目标 < 此值 = 放置成功
PLACED_Z_MAX = 0.05       # 物体 z < 此值 = 已落到地面（与 success 共同判定）

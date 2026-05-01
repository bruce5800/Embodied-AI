"""Phase-gated dense reward。

把 episode 自动切成两段：
  - 未抓起（held = False）：奖励"靠近 → 闭爪 → 接触 → 抬起"
  - 已抓起（held = True） ：奖励"运送到目标区"

held 判定 = 几何接触 AND 夹爪在闭合 AND 物体已离地。
"""

from dataclasses import dataclass

import numpy as np

from .scene_constants import (
    GRIPPER_IDX, HELD_GRIP_THRESHOLD,
    LIFT_THRESHOLD, LIFT_TARGET_Z,
    SUCCESS_RADIUS, PLACED_Z_MAX,
    OBJ_OOB_LOW, OBJ_OOB_HIGH,
    NEAR_GRIP_RADIUS,
)


# ─── 各项奖励权重 ───
# v2 改动：去掉 touch_bonus / near_grip_bonus（500k 训练显示 agent 用它们刷分
# 不真正抓起；正 reward + 0% success = 经典 reward gaming）。
# 新设计：phase A 只给 reach + lift（连续梯度），phase B 给明显更高的 baseline，
# 让 phase B 平均 reward ≫ phase A，agent 才有动力真抓起来。
W_ACTION_PENALTY = 0.001       # -‖a‖²
W_STEP_PENALTY = 0.01          # 每步小惩罚
W_LIFT_LINEAR = 5.0            # 抬起斜率（2 → 5：让 lift 变得显著有吸引力）
W_HELD_BASELINE = 3.0          # phase B 基础（1 → 3：远高于 phase A 平均）
W_OVERLIFT = 0.5               # 抬太高的扣分系数
R_PLACED_BONUS = 50.0          # 终止 bonus（20 → 50）


@dataclass
class RewardBreakdown:
    """每步 reward 各项明细，用于调试 / 日志。"""
    action_penalty: float
    step_penalty: float
    reach: float
    lift_linear: float
    held_baseline: float
    transport: float
    overlift_penalty: float
    placed_bonus: float
    total: float

    def to_dict(self):
        return self.__dict__.copy()


def compute_reward(
    data,
    *,
    site_ee_id: int,
    target_body_id: int,
    target_zone_pos: np.ndarray,
    has_contact: bool,
    action: np.ndarray,
):
    """
    单步 reward 计算。

    Args:
        data:               MjData（已 mj_forward）
        site_ee_id:         end_effector site id
        target_body_id:     当前 episode 的目标物体 body id
        target_zone_pos:    目标区中心 xyz
        has_contact:        gripper geom 是否接触 obj geom（env 用 mj_contact 算）
        action:             RL 动作（4-dim, 已裁剪到 [-1, 1]）

    Returns:
        reward (float)
        terminated (bool)         placed 成功
        truncated_oob (bool)      物体被推飞 ⇒ env 决定要不要 truncate
        info (dict)               调试信息
    """
    ee_pos = data.site_xpos[site_ee_id]
    obj_pos = data.xpos[target_body_id]
    target_pos = target_zone_pos

    obj_z = float(obj_pos[2])
    gripper_q = float(data.qpos[GRIPPER_IDX])

    # ── 状态判定 ──
    is_closing = gripper_q < HELD_GRIP_THRESHOLD
    is_lifted = obj_z > LIFT_THRESHOLD
    held = has_contact and is_closing and is_lifted

    # ── 距离指标 ──
    d_ee_obj = float(np.linalg.norm(obj_pos - ee_pos))
    d_obj_tgt_xy = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))

    # ── 终止 / OOB 判定 ──
    placed = (d_obj_tgt_xy < SUCCESS_RADIUS) and (obj_z < PLACED_Z_MAX)
    oob = bool(
        obj_pos[0] < OBJ_OOB_LOW[0] or obj_pos[0] > OBJ_OOB_HIGH[0] or
        obj_pos[1] < OBJ_OOB_LOW[1] or obj_pos[1] > OBJ_OOB_HIGH[1]
    )

    # ── 公共项 ──
    action_penalty = -W_ACTION_PENALTY * float(np.sum(np.asarray(action) ** 2))
    step_penalty = -W_STEP_PENALTY

    # ── 分阶段项 ──
    reach = 0.0
    lift_linear = 0.0
    held_baseline = 0.0
    transport = 0.0
    overlift_penalty = 0.0

    if not held:
        # 阶段 A：接近 + 抬起（移除了 touch/near_grip 重复 bonus）
        reach = -d_ee_obj
        lift_linear = W_LIFT_LINEAR * max(0.0, obj_z - LIFT_THRESHOLD)
    else:
        # 阶段 B：保持抓握并运送到目标区。baseline 高，明显优于 phase A
        held_baseline = W_HELD_BASELINE
        transport = -d_obj_tgt_xy
        overlift_penalty = -W_OVERLIFT * max(0.0, obj_z - LIFT_TARGET_Z)

    placed_bonus = R_PLACED_BONUS if placed else 0.0

    total = (action_penalty + step_penalty
             + reach + lift_linear
             + held_baseline + transport + overlift_penalty
             + placed_bonus)

    breakdown = RewardBreakdown(
        action_penalty=action_penalty,
        step_penalty=step_penalty,
        reach=reach,
        lift_linear=lift_linear,
        held_baseline=held_baseline,
        transport=transport,
        overlift_penalty=overlift_penalty,
        placed_bonus=placed_bonus,
        total=total,
    )

    info = {
        "held": held,
        "lifted": is_lifted,
        "contact": has_contact,
        "closing": is_closing,
        "placed": placed,
        "oob": oob,
        "d_ee_obj": d_ee_obj,
        "d_obj_tgt": d_obj_tgt_xy,
        "obj_z": obj_z,
        "reward_breakdown": breakdown.to_dict(),
    }

    return total, placed, oob, info

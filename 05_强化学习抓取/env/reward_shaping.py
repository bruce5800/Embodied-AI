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
# v3 改动（针对 500k 训练发现的 OOB hack）：
# 现象：mean_length 250→157，lift_rate 始终 ≤20%，agent 学撞飞物体提前 truncate。
# 根因：OOB 没惩罚，截断本身让 agent 少受 step_penalty + reach 负值，等价奖励逃跑。
# 修复：
#   1) OOB penalty −10：堵住"自杀逃跑"漏洞
#   2) reach clip 到 −1.0：防止远距离时 reach 项过负，让 OOB 显得诱人
#   3) 首次抬起 +5 一次性 bonus：鼓励真抓，但不能每步刷
W_ACTION_PENALTY = 0.001
W_STEP_PENALTY = 0.01
W_REACH_CLIP = 1.0             # |reach| ≤ 此值
W_LIFT_LINEAR = 5.0
W_HELD_BASELINE = 3.0
W_OVERLIFT = 0.5
W_TRANSPORT = 3.0              # phase B transport 系数（原本是 1.0 隐式，现显式加大）
R_FIRST_LIFT_BONUS = 20.0      # 首次抬起，episode 内一次性
R_FIRST_NEAR_TARGET = 15.0     # 首次抓着物体靠近目标 < 10cm，episode 内一次性
NEAR_TARGET_RADIUS = 0.10      # 触发 first_near_target 的距离阈值
R_PLACED_BONUS = 50.0
R_OOB_PENALTY = 10.0           # OOB 时给 −R_OOB_PENALTY


@dataclass
class RewardBreakdown:
    """每步 reward 各项明细，用于调试 / 日志。"""
    action_penalty: float
    step_penalty: float
    reach: float
    lift_linear: float
    first_lift_bonus: float
    held_baseline: float
    transport: float
    first_near_target_bonus: float
    overlift_penalty: float
    placed_bonus: float
    oob_penalty: float
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
    first_lift_pending: bool,
    first_near_target_pending: bool,
):
    """
    单步 reward 计算。

    Args:
        data:                MjData（已 mj_forward）
        site_ee_id:          end_effector site id
        target_body_id:      当前 episode 的目标物体 body id
        target_zone_pos:     目标区中心 xyz
        has_contact:         gripper geom 是否接触 obj geom（env 用 mj_contact 算）
        action:              RL 动作（已裁剪到 [-1, 1]）
        first_lift_pending:  episode 内还没发过 first_lift bonus（env 跟踪）

    Returns:
        reward (float)
        terminated (bool)         placed 成功
        truncated_oob (bool)      物体被推飞 ⇒ env 决定要不要 truncate
        info (dict)               调试信息（含 first_lift_consumed flag）
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
    first_lift_bonus = 0.0
    held_baseline = 0.0
    transport = 0.0
    first_near_target_bonus = 0.0
    overlift_penalty = 0.0

    # 一次性首次抬起 bonus
    first_lift_consumed = False
    if first_lift_pending and is_lifted:
        first_lift_bonus = R_FIRST_LIFT_BONUS
        first_lift_consumed = True

    if not held:
        # 阶段 A：接近 + 抬起。reach 加 clip 防极端值
        reach = -min(d_ee_obj, W_REACH_CLIP)
        lift_linear = W_LIFT_LINEAR * max(0.0, obj_z - LIFT_THRESHOLD)
    else:
        # 阶段 B：保持抓握并运送到目标区
        held_baseline = W_HELD_BASELINE
        transport = -W_TRANSPORT * d_obj_tgt_xy
        overlift_penalty = -W_OVERLIFT * max(0.0, obj_z - LIFT_TARGET_Z)

    # 一次性首次靠近目标 bonus（必须 held 状态下，并且 xy 距 < NEAR_TARGET_RADIUS）
    first_near_consumed = False
    if (first_near_target_pending and held
            and d_obj_tgt_xy < NEAR_TARGET_RADIUS):
        first_near_target_bonus = R_FIRST_NEAR_TARGET
        first_near_consumed = True

    placed_bonus = R_PLACED_BONUS if placed else 0.0
    # OOB 强惩罚：堵住"撞飞物体提前 truncate 逃跑"的 reward gaming
    oob_penalty = -R_OOB_PENALTY if oob else 0.0

    total = (action_penalty + step_penalty
             + reach + lift_linear + first_lift_bonus
             + held_baseline + transport + first_near_target_bonus
             + overlift_penalty
             + placed_bonus + oob_penalty)

    breakdown = RewardBreakdown(
        action_penalty=action_penalty,
        step_penalty=step_penalty,
        reach=reach,
        lift_linear=lift_linear,
        first_lift_bonus=first_lift_bonus,
        held_baseline=held_baseline,
        transport=transport,
        first_near_target_bonus=first_near_target_bonus,
        overlift_penalty=overlift_penalty,
        placed_bonus=placed_bonus,
        oob_penalty=oob_penalty,
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
        "first_lift_consumed": first_lift_consumed,
        "first_near_consumed": first_near_consumed,
        "reward_breakdown": breakdown.to_dict(),
    }

    return total, placed, oob, info

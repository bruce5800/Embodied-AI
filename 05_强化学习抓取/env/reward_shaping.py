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
W_REACH_CLIP = 1.0
W_LIFT_LINEAR = 5.0
# v9 修：v8 把 held_baseline 砍到 0 后 agent 学到"撞飞物体刷 first_lift +20"hack
# （OOB 58%, held 0%）。回到 0.5：phase B 有最低入场激励，但不像 v7 的 3.0 那样
# 让"抱着到超时"=600 >> placed_bonus。
# 数学：抱着 250 步累积 = 0.5×250 = 125 < placed_bonus 200，agent 选放下
W_HELD_BASELINE = 0.5
W_OVERLIFT = 0.5
# v6 回滚：v5 的 W_TRANSPORT=3 让 agent 急冲撞飞物体（OOB 30%），回到 1.0
W_TRANSPORT = 1.0          # 势能项：-W_TRANSPORT * d_obj_tgt_xy（保留作 phase B baseline）
# v10 加：progress reward —— 物体每步靠近 zone 多少（势能差），强化"运送"信号
# stage 2.5 (r=0.20) 退步暴露 transport 势能不够：agent 不知道"朝 zone 移动"是好的
W_PROGRESS = 10.0          # held 时每米靠近给 +10 reward，远离时 0（不罚，避免抖动）
R_FIRST_LIFT_BONUS = 20.0
# v6 回滚：v5 的 R_FIRST_NEAR_TARGET=15 reward 跳跃太大让 critic 学崩，
# 缩到 5.0 当温和 milestone
R_FIRST_NEAR_TARGET = 5.0
NEAR_TARGET_RADIUS = 0.10
# v8 修：50 → 200，让"放下成功"显著优于"抱着到超时"
R_PLACED_BONUS = 200.0
R_OOB_PENALTY = 10.0


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
    progress: float
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
    ever_lifted: bool,
    prev_d_obj_tgt: float,
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
    # v7 修：placed 必须 episode 内曾经 lift 过 —— 堵住"推杆刷分"漏洞
    # （v6 诊断显示 agent 学会用 gripper 推物体到 zone，从不抬起：
    #   placed 14%, held 0%, final obj_z 0.025m 这种数学不可能事件）
    placed = (
        ever_lifted
        and (d_obj_tgt_xy < SUCCESS_RADIUS)
        and (obj_z < PLACED_Z_MAX)
    )
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
    # v9 修：要求 held=True（不能光靠物体被撞飞触发）。否则 agent 学撞飞 hack
    first_lift_consumed = False
    if first_lift_pending and held:
        first_lift_bonus = R_FIRST_LIFT_BONUS
        first_lift_consumed = True

    progress = 0.0
    if not held:
        # 阶段 A：接近 + 抬起。reach 加 clip 防极端值
        reach = -min(d_ee_obj, W_REACH_CLIP)
        lift_linear = W_LIFT_LINEAR * max(0.0, obj_z - LIFT_THRESHOLD)
    else:
        # 阶段 B：保持抓握并运送到目标区
        held_baseline = W_HELD_BASELINE
        transport = -W_TRANSPORT * d_obj_tgt_xy
        # progress reward：朝 zone 靠近时给奖励（远离时不罚，避免抖动 hack）
        delta = prev_d_obj_tgt - d_obj_tgt_xy   # > 0 = 靠近
        progress = W_PROGRESS * max(0.0, delta)
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
             + held_baseline + transport + progress + first_near_target_bonus
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
        progress=progress,
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
        "d_obj_tgt_for_next": d_obj_tgt_xy,  # env 用作下一步的 prev_d_obj_tgt
        "obj_z": obj_z,
        "first_lift_consumed": first_lift_consumed,
        "first_near_consumed": first_near_consumed,
        "reward_breakdown": breakdown.to_dict(),
    }

    return total, placed, oob, info

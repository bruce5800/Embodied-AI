"""expert.py — 在 05 GraspEnv 里用 IK + scripted 5 阶段抓取-放置。

目的：
  1) 验证 05 env 的物理是否支持脚本式抓取（"修过的 KP/碰撞分组"是否够）
  2) 如果 ≥ 60% placed，可作为 BC 的 demo 数据来源
  3) 当 SAC 的对照基线（"经典 IK 比 RL 好/坏多少"）

实现：
  - IK 多重启（来自 04_/ik_solver.py）算关节绝对目标角
  - 转成 6-dim 关节增量动作：(target_q - cur_ctrl) / JOINT_DELTA_MAX，clip [-1, 1]
  - 5 阶段：pre_grasp → grasp → close → lift → transport → descend_place → release
  - 每阶段循环 step 直到 EE 到位 或 超时

跑：
  python expert.py --episodes 50
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import mujoco

from env import GraspEnv
from env.scene_constants import (
    N_ARM_JOINTS, READY_JOINTS, JOINT_DELTA_MAX,
)


def save_video(frames, path, fps=30):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


# ─── 抓取几何参数 ───
# 关键测量：EE site 在 gripper body 内部上方 9cm，gripper geom 最低点比 site 低 6.5cm
#   → site z = X 时，gripper finger 末端 z = X - 0.065
# 所以要让 gripper 末端在物体顶部（z=obj_z+物体半径），site 必须在物体中心上方 ≈ 7-8cm
PRE_GRASP_HEIGHT = 0.10
GRASP_Z_OFFSET = 0.06
                                           # → gripper finger 末端在物体中心 + 0.005
                                           # 之前 0.02 让 gripper 末端在地下 4.5cm，挤飞物体
LIFT_HEIGHT = 0.20
GRASP_XY_OFFSET = np.array([0.02, -0.02])  # 04 GRASP_OFFSET 的 xy 投影
PLACE_HEIGHT_LOW = 0.04                    # release 时 site 离 zone 4cm
                                           # 用户视频反馈穿地仍在，说明 6.5cm 估算偏小
                                           # release 用更低高度：finger 离 zone 更近，物体不会滚远

# ─── IK ───
IK_MAX_ITER = 200
IK_TOL = 1e-3
IK_DAMPING = 0.05
IK_NUM_ATTEMPTS = 8

# ─── Scripted timing ───
PHASE_MAX_ITERS = 150
GOTO_TOL = 0.025
# v3 实测：lift 57% 但 OOB 50%（抓住 17 个掉 15 个）
# 增加 GRIP_HOLD 让 gripper 充分关闭 + 物体稳定后再抬起
GRIP_HOLD_STEPS = 50        # 20 → 50

# v3 实测：GRIP_CLOSE -0.6 比 -0.4 close 失败更多（11 vs 7）。-0.4 (ctrl ≈ -3°) 是
# 最稳的配置——温和闭合，物体不被推开
GRIP_CLOSE_ACTION = -0.1
GRIP_OPEN_ACTION = 1.0

# v3 抬起阶段慢慢来：抬起前先让物体稳定几十步
LIFT_SETTLE_STEPS = 30      # 抓握后稳定步数（保持 ctrl 不变让物体被挤稳）


# 目标姿态：identity quat = ready 姿态时的 site frame，等价于 gripper 竖直朝下
# （ready 时 site z-axis = +world_z，gripper 张口朝 -site_z = -world_z 朝下）
TARGET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# IK 模式：
#   "3dof_full"     — 5 DOF IK 解 EE position；wrist 朝向自由
#   "3dof_locked"   — 锁 joint4/5 = ready 值（实测：30/30 卡 pre_grasp，因为锁
#                     joint4 仅限 link5 相对 link4 旋转，不保证 world 朝向；放弃）
#   "6dof_weighted" — POS+ROT 加权 6-DOF（实测：5 DOF 数学限制让 pos 变差）
# 当前选 3dof_full + 收紧 GRASP_OFFSET，依靠 collision groups 修复 (arm.ca=5)
# 阻止 arm 穿地板，让 gripper 必须从合理姿态接触物体
IK_MODE = "3dof_full"
POS_WEIGHT = 1.0
ROT_WEIGHT = 0.0


def solve_ik_locked_wrist(model, init_qpos, target_pos):
    """3-DOF IK with locked wrist。

    锁定 joint4/5 = READY_JOINTS 值（gripper 垂直朝下姿态），
    仅用 joint1-3 (base/shoulder/elbow) 解 EE position。
    3 DOF + 3 约束 = 完美匹配，no over-constraint。
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    target = np.asarray(target_pos, dtype=np.float64)
    eye3 = np.eye(3)

    locked_q3 = float(READY_JOINTS[3])  # wrist pitch
    locked_q4 = float(READY_JOINTS[4])  # wrist rotation

    best_q = None
    best_err = float("inf")

    for attempt in range(IK_NUM_ATTEMPTS):
        ik_data = mujoco.MjData(model)
        ik_data.qpos[:] = init_qpos[:]
        # 强制锁定 wrist
        ik_data.qpos[3] = locked_q3
        ik_data.qpos[4] = locked_q4
        if attempt == 1:
            # 第二次尝试从 ready 整体重启
            ik_data.qpos[:N_ARM_JOINTS] = READY_JOINTS
        elif attempt > 1:
            # 后续尝试只随机化 joint1-3，wrist 仍锁定
            for j in range(3):
                lo, hi = model.jnt_range[j]
                ik_data.qpos[j] = np.random.uniform(lo, hi)
            ik_data.qpos[3] = locked_q3
            ik_data.qpos[4] = locked_q4

        for _ in range(IK_MAX_ITER):
            mujoco.mj_forward(model, ik_data)
            err = target - ik_data.site_xpos[site_id]
            if np.linalg.norm(err) < IK_TOL:
                break
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, ik_data, jacp, None, site_id)
            J = jacp[:, :3]   # 仅取 joint1-3 列（3×3）
            try:
                dq = J.T @ np.linalg.solve(
                    J @ J.T + IK_DAMPING**2 * eye3, err
                )
            except np.linalg.LinAlgError:
                break
            scale = float(np.max(np.abs(dq)))
            if scale > 0.1:
                dq *= 0.1 / scale
            ik_data.qpos[:3] += dq  # 仅更新 joint1-3
            for j in range(3):
                lo, hi = model.jnt_range[j]
                ik_data.qpos[j] = np.clip(ik_data.qpos[j], lo, hi)
            # joint4/5 保持锁定（PD 漂移防御）
            ik_data.qpos[3] = locked_q3
            ik_data.qpos[4] = locked_q4

        mujoco.mj_forward(model, ik_data)
        final_err = float(np.linalg.norm(target - ik_data.site_xpos[site_id]))
        if final_err < best_err:
            best_err = final_err
            best_q = ik_data.qpos[:N_ARM_JOINTS].copy()
        if best_err < IK_TOL:
            break

    return best_q, best_err


def solve_ik_multistart(model, init_qpos, target_pos):
    """多重启 6-DOF DLS IK：约束 EE position + orientation（gripper 竖直朝下）。

    SO-100 是 5 DOF，6 个目标约束（3 pos + 3 rot）数学上不可能精确满足。
    用加权 DLS 求"位置精确 + 朝向尽量"的最优近似解。

    返回 (best_q[:5], err_pos)。err 只统计位置部分，因为我们更关心位置精度。
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    target_p = np.asarray(target_pos, dtype=np.float64)
    eye6 = np.eye(6)
    weights = np.array([POS_WEIGHT]*3 + [ROT_WEIGHT]*3, dtype=np.float64)

    best_q = None
    best_err = float("inf")

    for attempt in range(IK_NUM_ATTEMPTS):
        ik_data = mujoco.MjData(model)
        ik_data.qpos[:] = init_qpos[:]
        if attempt == 1:
            ik_data.qpos[:N_ARM_JOINTS] = READY_JOINTS
        elif attempt > 1:
            for j in range(N_ARM_JOINTS):
                lo, hi = model.jnt_range[j]
                ik_data.qpos[j] = np.random.uniform(lo, hi)

        for _ in range(IK_MAX_ITER):
            mujoco.mj_forward(model, ik_data)

            # ── position error ──
            pos_err = target_p - ik_data.site_xpos[site_id]

            # ── orientation error: target × inv(cur) → axis-angle ──
            cur_quat = np.zeros(4)
            mujoco.mju_mat2Quat(cur_quat, ik_data.site_xmat[site_id])
            neg_cur = cur_quat.copy()
            neg_cur[1:] *= -1   # 单位四元数的逆 = 共轭
            err_quat = np.zeros(4)
            mujoco.mju_mulQuat(err_quat, TARGET_QUAT, neg_cur)
            rot_err = np.zeros(3)
            mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)  # axis × angle

            err_6 = np.concatenate([pos_err, rot_err])
            err_6w = err_6 * weights
            if np.linalg.norm(pos_err) < IK_TOL and np.linalg.norm(rot_err) < 0.1:
                break

            # ── jacobian: 3×nv pos + 3×nv rot ──
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
            J = np.vstack([jacp[:, :N_ARM_JOINTS],
                          jacr[:, :N_ARM_JOINTS]])    # 6×5
            Jw = J * weights[:, None]

            try:
                dq = Jw.T @ np.linalg.solve(
                    Jw @ Jw.T + IK_DAMPING**2 * eye6, err_6w
                )
            except np.linalg.LinAlgError:
                break

            scale = float(np.max(np.abs(dq)))
            if scale > 0.1:
                dq *= 0.1 / scale

            ik_data.qpos[:N_ARM_JOINTS] += dq
            for j in range(N_ARM_JOINTS):
                lo, hi = model.jnt_range[j]
                ik_data.qpos[j] = np.clip(ik_data.qpos[j], lo, hi)

        mujoco.mj_forward(model, ik_data)
        final_pos_err = float(np.linalg.norm(target_p - ik_data.site_xpos[site_id]))
        if final_pos_err < best_err:
            best_err = final_pos_err
            best_q = ik_data.qpos[:N_ARM_JOINTS].copy()
        if best_err < IK_TOL:
            break

    return best_q, best_err


def joint_action(target_q, cur_ctrl, gripper_open):
    """绝对关节目标 → 6-dim 增量动作 [-1, 1]。"""
    delta = target_q - cur_ctrl
    a_arm = np.clip(delta / JOINT_DELTA_MAX, -1.0, 1.0)
    a_grip = GRIP_OPEN_ACTION if gripper_open else GRIP_CLOSE_ACTION
    return np.concatenate([a_arm, [a_grip]]).astype(np.float32)


def run_episode(env, seed, verbose=False, record_frames=False):
    """跑一个 episode，返回结果 dict。

    record_frames=True 时会在每步调用 env.render() 收集帧；要求 env 创建时
    render_mode='rgb_array'。
    """
    obs, info = env.reset(seed=seed)
    obj_pos = info["object_pos"].copy()
    target_zone = info["target_zone"].copy()
    frames = [] if record_frames else None

    # waypoints
    grasp_xy = obj_pos[:2] + GRASP_XY_OFFSET
    grasp_pos = np.array([grasp_xy[0], grasp_xy[1], obj_pos[2] + GRASP_Z_OFFSET])
    pre_grasp = grasp_pos + np.array([0, 0, PRE_GRASP_HEIGHT])
    lift_pos = grasp_pos + np.array([0, 0, LIFT_HEIGHT])
    place_above = np.array([target_zone[0], target_zone[1], LIFT_HEIGHT])
    place_low = np.array([target_zone[0], target_zone[1], PLACE_HEIGHT_LOW])

    state = {"info": info, "term": False, "trunc": False, "ever_lifted": False}

    def _maybe_record():
        if frames is not None:
            frames.append(env.render())

    def goto(target_pos, gripper_open, label):
        """循环走到 target_pos。"""
        for _ in range(PHASE_MAX_ITERS):
            data = env._data
            if IK_MODE == "3dof_locked":
                target_q, _ = solve_ik_locked_wrist(env._model, data.qpos, target_pos)
            else:
                target_q, _ = solve_ik_multistart(env._model, data.qpos, target_pos)
            if target_q is None:
                return False
            cur_ctrl = data.ctrl[:N_ARM_JOINTS].copy()
            action = joint_action(target_q, cur_ctrl, gripper_open)
            _, _, state["term"], state["trunc"], state["info"] = env.step(action)
            _maybe_record()
            state["ever_lifted"] = (state["ever_lifted"]
                                    or state["info"].get("lifted", False))
            ee = state["info"]["ee_pos"]
            if np.linalg.norm(ee - target_pos) < GOTO_TOL:
                return True
            if state["term"] or state["trunc"]:
                return False
        return False  # 超时

    def hold_gripper(close):
        """保持当前关节位置，只动 gripper。"""
        for _ in range(GRIP_HOLD_STEPS):
            action = np.zeros(6, dtype=np.float32)
            action[5] = GRIP_CLOSE_ACTION if close else GRIP_OPEN_ACTION
            _, _, state["term"], state["trunc"], state["info"] = env.step(action)
            _maybe_record()
            state["ever_lifted"] = (state["ever_lifted"]
                                    or state["info"].get("lifted", False))
            if state["term"] or state["trunc"]:
                return False
        return True

    def settle(n_steps):
        """保持当前 ctrl + 闭爪不变，让物体在 gripper 内稳定。"""
        for _ in range(n_steps):
            action = np.zeros(6, dtype=np.float32)
            action[5] = GRIP_CLOSE_ACTION
            _, _, state["term"], state["trunc"], state["info"] = env.step(action)
            _maybe_record()
            state["ever_lifted"] = (state["ever_lifted"]
                                    or state["info"].get("lifted", False))
            if state["term"] or state["trunc"]:
                return False
        return True

    phase_failed = None
    if not goto(pre_grasp, True, "pre_grasp"):
        phase_failed = "pre_grasp"
    elif not goto(grasp_pos, True, "grasp"):
        phase_failed = "descend_grasp"
    elif not hold_gripper(close=True):
        phase_failed = "close"
    elif not settle(LIFT_SETTLE_STEPS):  # 让 gripper 完全闭合稳住物体后再抬
        phase_failed = "settle"
    elif not goto(lift_pos, False, "lift"):
        phase_failed = "lift"
    elif not goto(place_above, False, "transport"):
        phase_failed = "transport"
    elif not goto(place_low, False, "descend_place"):
        phase_failed = "descend_place"
    elif not hold_gripper(close=False):
        phase_failed = "release"

    return {
        "placed": bool(state["info"].get("placed", False)),
        "lifted": bool(state["ever_lifted"]),
        "oob": bool(state["info"].get("oob", False)),
        "steps": int(state["info"].get("steps", 0)),
        "phase_failed": phase_failed,
        "obj_pos_init": obj_pos,
        "frames": frames,  # None 或 list of RGB frames
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--target", default="red_cylinder",
                        help="目标物体（'random' 表示随机）")
    parser.add_argument("--max-episode-steps", type=int, default=1200,
                        help="env 内 max_episode_steps（脚本 6 阶段 × 150 iter ≈ 900 步）")
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--render", action="store_true",
                        help="录视频保存到 expert_renders/")
    parser.add_argument("--n-videos", type=int, default=10,
                        help="录前 N 个 episode（含成功 + 失败）")
    args = parser.parse_args()

    target = None if args.target == "random" else args.target
    render_mode = "rgb_array" if args.render else None
    env = GraspEnv(target_object=target, max_episode_steps=args.max_episode_steps,
                   render_mode=render_mode)

    render_dir = Path(__file__).resolve().parent / "expert_renders"
    if args.render:
        render_dir.mkdir(exist_ok=True)

    results = []
    fail_counter = Counter()

    for ep in range(args.episodes):
        record = args.render and ep < args.n_videos
        r = run_episode(env, seed=args.seed_base + ep, record_frames=record)
        results.append(r)
        if r["phase_failed"] is not None:
            fail_counter[r["phase_failed"]] += 1
        marker = "✓" if r["placed"] else ("L" if r["lifted"] else "✗")
        oob_mark = " OOB" if r["oob"] else ""

        # 保存视频
        if record and r["frames"]:
            tag = "OK" if r["placed"] else ("LIFT" if r["lifted"] else "FAIL")
            fail_str = f"_{r['phase_failed']}" if r['phase_failed'] else ""
            path = render_dir / f"ep{ep:03d}_{tag}{fail_str}.mp4"
            save_video(r["frames"], path)
            video_str = f" → {path.name}"
        else:
            video_str = ""

        print(f"ep {ep:3d} [{marker}]: "
              f"lift={int(r['lifted'])} place={int(r['placed'])} "
              f"steps={r['steps']:3d} fail={r['phase_failed'] or '-':14s}"
              f"{oob_mark}{video_str}")

    n = args.episodes
    n_placed = sum(r["placed"] for r in results)
    n_lifted = sum(r["lifted"] for r in results)
    n_oob = sum(r["oob"] for r in results)
    avg_steps = np.mean([r["steps"] for r in results])

    print("─" * 60)
    print(f"  episodes      : {n}")
    print(f"  placed        : {n_placed/n:.0%}  ({n_placed}/{n})")
    print(f"  lifted (ever) : {n_lifted/n:.0%}  ({n_lifted}/{n})")
    print(f"  oob           : {n_oob/n:.0%}   ({n_oob}/{n})")
    print(f"  avg steps     : {avg_steps:.0f}")
    print()
    if fail_counter:
        print("  阶段失败分布（按出现次数排序）:")
        for phase, cnt in fail_counter.most_common():
            print(f"    {phase:20s} : {cnt}")

    env.close()


if __name__ == "__main__":
    main()

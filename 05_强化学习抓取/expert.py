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

import numpy as np
import mujoco

from env import GraspEnv
from env.scene_constants import (
    N_ARM_JOINTS, READY_JOINTS, JOINT_DELTA_MAX,
)


# ─── 抓取几何参数（先用 04 经验值，跑完按失败模式调） ───
PRE_GRASP_HEIGHT = 0.10
GRASP_Z_OFFSET = 0.03                      # EE 高于物体中心
LIFT_HEIGHT = 0.20
GRASP_XY_OFFSET = np.array([0.02, -0.02])  # 04 GRASP_OFFSET 的 xy 投影
PLACE_HEIGHT_LOW = 0.04                    # 松手时 EE 离 zone 中心高度

# ─── IK ───
IK_MAX_ITER = 200
IK_TOL = 1e-3
IK_DAMPING = 0.05
IK_NUM_ATTEMPTS = 8

# ─── Scripted timing ───
PHASE_MAX_ITERS = 80
GOTO_TOL = 0.015            # EE 接近 target 视为到位
GRIP_HOLD_STEPS = 15        # 闭/张夹爪保持步数


def solve_ik_multistart(model, init_qpos, target_pos):
    """多重启 DLS IK，返回 (best_q[:5], err)。"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    target = np.asarray(target_pos, dtype=np.float64)
    eye3 = np.eye(3)

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
            err = target - ik_data.site_xpos[site_id]
            if np.linalg.norm(err) < IK_TOL:
                break
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, ik_data, jacp, None, site_id)
            J = jacp[:, :N_ARM_JOINTS]
            try:
                dq = J.T @ np.linalg.solve(
                    J @ J.T + IK_DAMPING**2 * eye3, err
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
        final_err = float(np.linalg.norm(target - ik_data.site_xpos[site_id]))
        if final_err < best_err:
            best_err = final_err
            best_q = ik_data.qpos[:N_ARM_JOINTS].copy()
        if best_err < IK_TOL:
            break

    return best_q, best_err


def joint_action(target_q, cur_ctrl, gripper_open):
    """绝对关节目标 → 6-dim 增量动作 [-1, 1]。"""
    delta = target_q - cur_ctrl
    a_arm = np.clip(delta / JOINT_DELTA_MAX, -1.0, 1.0)
    a_grip = 1.0 if gripper_open else -1.0
    return np.concatenate([a_arm, [a_grip]]).astype(np.float32)


def run_episode(env, seed, verbose=False):
    """跑一个 episode，返回结果 dict。"""
    obs, info = env.reset(seed=seed)
    obj_pos = info["object_pos"].copy()
    target_zone = info["target_zone"].copy()

    # waypoints
    grasp_xy = obj_pos[:2] + GRASP_XY_OFFSET
    grasp_pos = np.array([grasp_xy[0], grasp_xy[1], obj_pos[2] + GRASP_Z_OFFSET])
    pre_grasp = grasp_pos + np.array([0, 0, PRE_GRASP_HEIGHT])
    lift_pos = grasp_pos + np.array([0, 0, LIFT_HEIGHT])
    place_above = np.array([target_zone[0], target_zone[1], LIFT_HEIGHT])
    place_low = np.array([target_zone[0], target_zone[1], PLACE_HEIGHT_LOW])

    state = {"info": info, "term": False, "trunc": False, "ever_lifted": False}

    def goto(target_pos, gripper_open, label):
        """循环走到 target_pos。"""
        for _ in range(PHASE_MAX_ITERS):
            data = env._data
            target_q, _ = solve_ik_multistart(env._model, data.qpos, target_pos)
            if target_q is None:
                return False
            cur_ctrl = data.ctrl[:N_ARM_JOINTS].copy()
            action = joint_action(target_q, cur_ctrl, gripper_open)
            _, _, state["term"], state["trunc"], state["info"] = env.step(action)
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
            action[5] = -1.0 if close else 1.0
            _, _, state["term"], state["trunc"], state["info"] = env.step(action)
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--target", default="red_cylinder",
                        help="目标物体（'random' 表示随机）")
    parser.add_argument("--max-episode-steps", type=int, default=600,
                        help="env 内 max_episode_steps（脚本至少需 ~500）")
    parser.add_argument("--seed-base", type=int, default=10_000)
    args = parser.parse_args()

    target = None if args.target == "random" else args.target
    env = GraspEnv(target_object=target, max_episode_steps=args.max_episode_steps)

    results = []
    fail_counter = Counter()

    for ep in range(args.episodes):
        r = run_episode(env, seed=args.seed_base + ep)
        results.append(r)
        if r["phase_failed"] is not None:
            fail_counter[r["phase_failed"]] += 1
        marker = "✓" if r["placed"] else ("L" if r["lifted"] else "✗")
        oob_mark = " OOB" if r["oob"] else ""
        print(f"ep {ep:3d} [{marker}]: "
              f"lift={int(r['lifted'])} place={int(r['placed'])} "
              f"steps={r['steps']:3d} fail={r['phase_failed'] or '-':14s}"
              f"{oob_mark}")

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

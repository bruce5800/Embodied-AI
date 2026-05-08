"""诊断 agent 行为：接近 / 闭爪 / 接触 / 抬起 各阶段是否发生。

用法：
    python diagnose.py --ckpt checkpoints/m1_v3/best_model.zip
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
from stable_baselines3 import SAC

from env import GraspEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--target", default="red_cylinder")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--obj-radius", type=float, default=None,
                        help="跟训练时同样的 curriculum_radius，否则评估不可比")
    args = parser.parse_args()

    target = None if args.target == "random" else args.target
    env = GraspEnv(target_object=target, max_episode_steps=args.max_steps,
                   curriculum_radius=args.obj_radius)
    model = SAC.load(args.ckpt, env=env)

    # 阶段计数（per episode）
    stats = {
        "approached":     0,   # episode 中 d_ee_obj 曾 < 0.05
        "contacted":      0,   # episode 中 contact_flag 曾 = 1
        "tried_close":    0,   # episode 中 gripper_q < HELD_GRIP_THRESHOLD 出现过
        "lifted":         0,   # obj_z 曾 > 0.06（含瞬间触发假阳性）
        "held":           0,   # 同时满足 contact + closing + lifted（任一帧）
        "stably_held":    0,   # held=True **连续 ≥ STABLE_HELD_STEPS 帧**（真抓握）
        "placed":         0,   # placed (含可能的 push hack)
        "placed_via_held":0,   # placed=True AND 之前曾 stably_held（真"抓-放"流程）
        "oob":            0,
    }
    STABLE_HELD_STEPS = 10  # 连续 10 步握住才算"稳定抓"

    min_dists = []
    contact_steps = []
    closing_steps = []
    final_obj_z = []
    held_max_runs = []   # 每 episode 最长连续 held 步数

    for ep in range(args.episodes):
        obs, info = env.reset(seed=10_000 + ep)
        ep_min_dist = float("inf")
        ep_contact = 0
        ep_close = 0
        ep_approached = ep_contacted = ep_tried_close = ep_lifted = ep_held = ep_oob = False

        # 跟踪 held 连续步数
        held_run = 0          # 当前连续 held 步数
        held_max = 0          # 最长连续 held
        ep_stably_held = False  # 是否曾达到 stably_held
        stably_held_before_placed = False  # placed 时是否之前已 stably_held

        for t in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)

            d = info["d_ee_obj"]
            ep_min_dist = min(ep_min_dist, d)
            if d < 0.05:
                ep_approached = True
            if info["contact"]:
                ep_contacted = True
                ep_contact += 1
            if info["closing"]:
                ep_tried_close = True
                ep_close += 1
            if info["lifted"]:
                ep_lifted = True
            if info["held"]:
                ep_held = True
                held_run += 1
                held_max = max(held_max, held_run)
                if held_run >= STABLE_HELD_STEPS:
                    ep_stably_held = True
            else:
                held_run = 0
            # 记录"placed 时是否之前已 stably_held"
            if info.get("placed", False) and ep_stably_held:
                stably_held_before_placed = True
            if info["oob"]:
                ep_oob = True
            if term or trunc:
                break

        stats["approached"]      += int(ep_approached)
        stats["contacted"]       += int(ep_contacted)
        stats["tried_close"]     += int(ep_tried_close)
        stats["lifted"]          += int(ep_lifted)
        stats["held"]            += int(ep_held)
        stats["stably_held"]     += int(ep_stably_held)
        stats["placed"]          += int(info.get("placed", False))
        stats["placed_via_held"] += int(stably_held_before_placed)
        stats["oob"]             += int(ep_oob)

        min_dists.append(ep_min_dist)
        contact_steps.append(ep_contact)
        closing_steps.append(ep_close)
        final_obj_z.append(info["obj_z"])
        held_max_runs.append(held_max)

    n = args.episodes
    print(f"\n=== {n} episodes, target={args.target} ===")
    print(f"  approached (d_ee_obj < 5cm)     : {stats['approached']/n:.0%}")
    print(f"  tried_close (gripper closed)    : {stats['tried_close']/n:.0%}")
    print(f"  contacted (gripper-obj contact) : {stats['contacted']/n:.0%}")
    print(f"  lifted    (obj_z > 6cm)         : {stats['lifted']/n:.0%}")
    print(f"  held      (any 1 step)          : {stats['held']/n:.0%}")
    print(f"  stably_held (≥{STABLE_HELD_STEPS} consecutive) : "
          f"{stats['stably_held']/n:.0%}  ← 真抓握判定")
    print(f"  placed    (in zone)             : {stats['placed']/n:.0%}")
    print(f"  placed_via_held (真抓-放成功)    : {stats['placed_via_held']/n:.0%}")
    print(f"  oob       (obj knocked away)    : {stats['oob']/n:.0%}")
    print()
    print(f"  min dist EE-obj  : avg {np.mean(min_dists):.3f}m  min {np.min(min_dists):.3f}m")
    print(f"  steps in contact : avg {np.mean(contact_steps):.1f}")
    print(f"  steps closing    : avg {np.mean(closing_steps):.1f}")
    print(f"  held_run_max     : avg {np.mean(held_max_runs):.1f}  "
          f"max {np.max(held_max_runs):.0f}")
    print(f"  final obj_z      : avg {np.mean(final_obj_z):.3f}m")

    env.close()


if __name__ == "__main__":
    main()

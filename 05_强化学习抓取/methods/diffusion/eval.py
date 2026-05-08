"""eval_diffusion.py — Diffusion Policy 评估 + 严格诊断 + 录视频。

用法：
    python eval_diffusion.py --ckpt checkpoints/dp_blue_v1/ema.pt \\
        --target blue_cube --episodes 50 --render
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

import cv2
import numpy as np

from diffusion_policy import DiffusionPolicy
from env import GraspEnv


REPO_ROOT = Path(__file__).resolve().parents[2]   # 项目根
STABLE_HELD_STEPS = 10


def save_video(frames, path, fps=30):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="ema.pt 路径")
    parser.add_argument("--target", default="blue_cube")
    parser.add_argument("--obj-radius", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--render", action="store_true",
                        help="录前 5 个 episode 视频")
    parser.add_argument("--n-render", type=int, default=5)
    parser.add_argument("--n-action-steps", type=int, default=4,
                        help="receding horizon: 每 N 步重新预测")
    parser.add_argument("--num-inference-steps", type=int, default=10,
                        help="DDIM 推理步数")
    parser.add_argument("--device", default="cpu",
                        help="推理设备（cpu/mps），cpu 通常够")
    parser.add_argument("--soften-contacts", action="store_true",
                        help="跟训练 demo 一致 (true 跟 collect_demos 默认开启)")
    parser.add_argument("--seed-base", type=int, default=10_000)
    args = parser.parse_args()

    target = None if args.target == "random" else args.target
    env = GraspEnv(
        target_object=target,
        max_episode_steps=args.max_steps,
        curriculum_radius=args.obj_radius,
        render_mode="rgb_array" if args.render else None,
        soften_contacts=args.soften_contacts,
    )

    policy = DiffusionPolicy(
        ckpt_path=args.ckpt,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        n_action_steps=args.n_action_steps,
    )

    render_dir = REPO_ROOT / "eval_renders" / "diffusion"
    if args.render:
        render_dir.mkdir(parents=True, exist_ok=True)

    # 严格判定累计
    stats = {k: 0 for k in [
        "approached", "tried_close", "contacted",
        "lifted", "held", "stably_held",
        "placed", "placed_via_held", "oob",
    ]}
    rewards, lengths = [], []
    held_max_runs = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed_base + ep)
        policy.reset()
        ep_reward, ep_len = 0.0, 0
        frames = [] if (args.render and ep < args.n_render) else None

        held_run = 0
        held_max = 0
        ep_stably_held = False
        placed_via_held = False
        flags = {k: False for k in stats}

        for t in range(args.max_steps):
            action = policy.predict(obs)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            ep_len += 1
            if frames is not None:
                frames.append(env.render())

            if info["d_ee_obj"] < 0.05: flags["approached"] = True
            if info["closing"]:         flags["tried_close"] = True
            if info["contact"]:         flags["contacted"] = True
            if info["lifted"]:          flags["lifted"] = True
            if info["held"]:
                flags["held"] = True
                held_run += 1
                held_max = max(held_max, held_run)
                if held_run >= STABLE_HELD_STEPS:
                    ep_stably_held = True
            else:
                held_run = 0
            if info.get("placed", False) and ep_stably_held:
                placed_via_held = True
            if info["oob"]: flags["oob"] = True
            if term or trunc:
                break

        if info.get("placed", False):  flags["placed"] = True
        flags["stably_held"] = ep_stably_held
        flags["placed_via_held"] = placed_via_held

        for k, v in flags.items():
            stats[k] += int(v)
        rewards.append(ep_reward)
        lengths.append(ep_len)
        held_max_runs.append(held_max)

        marker = "✓" if placed_via_held else (
            "P" if flags["placed"] else (
                "H" if ep_stably_held else (
                    "L" if flags["lifted"] else "✗")))
        print(f"  ep {ep:3d} [{marker}] r={ep_reward:+7.2f} len={ep_len:3d}  "
              f"lift={int(flags['lifted'])} held={int(ep_stably_held)} "
              f"placed={int(flags['placed'])} placed_via={int(placed_via_held)}")

        if frames is not None:
            tag = ("REAL_OK" if placed_via_held else
                   "PUSH_HACK" if flags["placed"] else
                   "HELD" if ep_stably_held else
                   "LIFT" if flags["lifted"] else "FAIL")
            path = render_dir / f"dp_ep{ep:03d}_{tag}.mp4"
            save_video(frames, path)

    n = args.episodes
    print(f"\n=== {n} episodes, target={args.target} ===")
    print(f"  approached            : {stats['approached']/n:.0%}")
    print(f"  tried_close           : {stats['tried_close']/n:.0%}")
    print(f"  contacted             : {stats['contacted']/n:.0%}")
    print(f"  lifted                : {stats['lifted']/n:.0%}")
    print(f"  held (any 1 step)     : {stats['held']/n:.0%}")
    print(f"  stably_held (≥10 step): {stats['stably_held']/n:.0%}  ← 真抓")
    print(f"  placed (in zone)      : {stats['placed']/n:.0%}")
    print(f"  placed_via_held       : {stats['placed_via_held']/n:.0%}  ← 真抓-放")
    print(f"  oob                   : {stats['oob']/n:.0%}")
    print()
    print(f"  mean reward          : {np.mean(rewards):+.2f}")
    print(f"  mean length          : {np.mean(lengths):.1f}")
    print(f"  held_run_max avg     : {np.mean(held_max_runs):.1f}")

    env.close()


if __name__ == "__main__":
    main()

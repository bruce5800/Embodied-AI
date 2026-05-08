"""录"假阳性 placed"视频：placed=True 但 episode 中**没有** stably_held。
即机械臂用张开 gripper 推/捞物体到 zone，没真抓握过。
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import cv2
import numpy as np
from stable_baselines3 import SAC
from env import GraspEnv

REPO_ROOT = Path(__file__).resolve().parents[1]   # 项目根
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
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--target", default="yellow_cylinder")
    parser.add_argument("--obj-radius", type=float, default=0.10)
    parser.add_argument("--n-target", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--out-dir", default="demos_videos")
    parser.add_argument("--prefix", default="push_hack")
    args = parser.parse_args()

    target = None if args.target == "random" else args.target
    env = GraspEnv(
        target_object=target, max_episode_steps=250,
        curriculum_radius=args.obj_radius, render_mode="rgb_array",
    )
    model = SAC.load(args.ckpt, env=env)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)

    found = 0
    for ep in range(args.max_attempts):
        obs, info = env.reset(seed=10_000 + ep)
        frames = []
        held_run = 0
        ever_stably_held = False
        for t in range(250):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            frames.append(env.render())
            if info["held"]:
                held_run += 1
                if held_run >= STABLE_HELD_STEPS:
                    ever_stably_held = True
            else:
                held_run = 0
            if term or trunc:
                break

        placed = info.get("placed", False)
        # push hack: placed 但没真抓
        if placed and not ever_stably_held:
            found += 1
            path = out_dir / f"{args.prefix}_{found}_seed{10_000+ep}.mp4"
            save_video(frames, path)
            print(f"  ep {ep} PUSH HACK (placed without stable held) → {path.name}")
            if found >= args.n_target:
                break

    print(f"\nCollected {found} push-hack videos")


if __name__ == "__main__":
    main()

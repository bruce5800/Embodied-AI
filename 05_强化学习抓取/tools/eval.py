"""加载 SAC checkpoint，评估 N episode，可选保存视频。

用法：
    # 只跑数字
    python eval.py --ckpt checkpoints/m1_red/best_model.zip

    # 同时保存前 5 个 episode 的视频
    python eval.py --ckpt checkpoints/m1_red/best_model.zip --render

    # 评估多物体随机 ckpt
    python eval.py --ckpt checkpoints/m2_random/best_model.zip --target random
"""


from __future__ import annotations

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


def save_video(frames, path, fps=30):
    """用 cv2 写 mp4。frames 是 RGB ndarray 列表。"""
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
    parser.add_argument("--ckpt", required=True, help="SAC 模型路径（.zip）")
    parser.add_argument("--target", default="red_cylinder",
                        help="目标物体；'random' 表示每 episode 随机")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--render", action="store_true",
                        help="保存前 N_RENDER 个 episode 视频到 eval_renders/")
    parser.add_argument("--n-render", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=10_000)
    args = parser.parse_args()

    target = None if args.target == "random" else args.target

    env = GraspEnv(
        target_object=target,
        max_episode_steps=args.max_steps,
        render_mode="rgb_array" if args.render else None,
    )
    model = SAC.load(args.ckpt, env=env)

    render_dir = REPO_ROOT / "eval_renders"
    if args.render:
        render_dir.mkdir(exist_ok=True)

    n_success = 0
    n_lifted = 0
    rewards = []
    lengths = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed_base + ep)
        ep_reward = 0.0
        ep_lifted = False
        frames = [] if (args.render and ep < args.n_render) else None

        for t in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            ep_lifted = ep_lifted or info.get("lifted", False)
            if frames is not None:
                frames.append(env.render())
            if term or trunc:
                break

        placed = info.get("placed", False)
        if placed:
            n_success += 1
        if ep_lifted:
            n_lifted += 1
        rewards.append(ep_reward)
        lengths.append(t + 1)

        if frames is not None:
            tag = "OK" if placed else ("LIFT" if ep_lifted else "FAIL")
            path = render_dir / f"ep{ep:03d}_{info['target_object']}_{tag}.mp4"
            save_video(frames, path)
            print(f"  saved {path}")

        print(f"ep {ep:3d}: r={ep_reward:+7.2f} len={t+1:3d} "
              f"target={info['target_object']:16s} "
              f"lift={'Y' if ep_lifted else 'N'} "
              f"place={'Y' if placed else 'N'}")

    print("─" * 60)
    print(f"  episodes      : {args.episodes}")
    print(f"  success_rate  : {n_success / args.episodes:.2%} "
          f"({n_success}/{args.episodes})")
    print(f"  lift_rate     : {n_lifted / args.episodes:.2%} "
          f"({n_lifted}/{args.episodes})")
    print(f"  mean_reward   : {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
    print(f"  mean_length   : {np.mean(lengths):.1f}")

    env.close()


if __name__ == "__main__":
    main()

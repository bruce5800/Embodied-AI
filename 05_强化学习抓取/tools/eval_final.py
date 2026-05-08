"""eval_final.py — 最终最佳 ckpt 综合评估 + 视频。

跑两组评估对比：
  1) 训练分布内（radius=0.10）：agent 训练时见过的物体范围
  2) zero-shot 全场景（radius=None）：泛化能力测试

各跑 100 episode，输出详细统计 + 录前 5 个视频。

默认 ckpt：v10 stage 2 best_model（当前最佳 placed 36%）

用法：
    python eval_final.py                                    # 用默认 ckpt
    python eval_final.py --ckpt path/to/other.zip           # 指定 ckpt
    python eval_final.py --episodes 50                      # 减少 episode 加速
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3 import SAC

from env import GraspEnv


REPO_ROOT = Path(__file__).resolve().parents[1]   # 项目根
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "sac" / "m1_yellow_s2_v3_buf" / "best_model.zip"


def save_video(frames, path, fps=30):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def run_eval(model, env, episodes, label, render_dir=None, n_videos=5):
    """跑 episodes 评估，返回 stats dict。"""
    stats = {
        "approached": 0, "tried_close": 0, "contacted": 0,
        "lifted": 0, "held": 0, "placed": 0, "oob": 0,
    }
    rewards, lengths = [], []
    min_dists, final_zs = [], []

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for ep in range(episodes):
        obs, info = env.reset(seed=20_000 + ep)
        ep_reward = 0.0
        ep_len = 0
        ep_min_dist = float("inf")
        flags = {k: False for k in stats}
        frames = [] if (render_dir is not None and ep < n_videos) else None

        for t in range(env._max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            ep_len += 1
            ep_min_dist = min(ep_min_dist, info["d_ee_obj"])

            if info["d_ee_obj"] < 0.05:   flags["approached"] = True
            if info["closing"]:           flags["tried_close"] = True
            if info["contact"]:           flags["contacted"] = True
            if info["lifted"]:            flags["lifted"] = True
            if info["held"]:              flags["held"] = True
            if info["oob"]:               flags["oob"] = True

            if frames is not None:
                frames.append(env.render())
            if term or trunc:
                break

        if info.get("placed", False):
            flags["placed"] = True

        for k, v in flags.items():
            stats[k] += int(v)
        rewards.append(ep_reward)
        lengths.append(ep_len)
        min_dists.append(ep_min_dist)
        final_zs.append(info["obj_z"])

        marker = "✓" if flags["placed"] else ("L" if flags["lifted"] else "✗")
        if frames is not None:
            tag = "OK" if flags["placed"] else ("LIFT" if flags["lifted"] else "FAIL")
            path = render_dir / f"{label}_ep{ep:03d}_{tag}.mp4"
            save_video(frames, path)
            print(f"  ep {ep:3d} [{marker}] reward={ep_reward:+7.2f} len={ep_len:3d}  "
                  f"saved {path.name}")
        else:
            print(f"  ep {ep:3d} [{marker}] reward={ep_reward:+7.2f} len={ep_len:3d}")

    n = episodes
    print(f"\n{label} 总结:")
    print(f"  approached   : {stats['approached']/n:.0%}")
    print(f"  tried_close  : {stats['tried_close']/n:.0%}")
    print(f"  contacted    : {stats['contacted']/n:.0%}")
    print(f"  lifted       : {stats['lifted']/n:.0%}")
    print(f"  held         : {stats['held']/n:.0%}")
    print(f"  placed       : {stats['placed']/n:.0%}  ({stats['placed']}/{n})")
    print(f"  oob          : {stats['oob']/n:.0%}")
    print(f"  mean reward  : {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
    print(f"  mean length  : {np.mean(lengths):.1f}")
    print(f"  mean min_dist: {np.mean(min_dists):.3f}m")
    print(f"  mean final_z : {np.mean(final_zs):.3f}m")

    return {
        "label": label,
        "n": n,
        "stats": stats,
        "rewards": rewards,
        "lengths": lengths,
        "min_dists": min_dists,
        "final_zs": final_zs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--target", default="yellow_cylinder")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--n-videos", type=int, default=5,
                        help="每组评估录的视频数（前 N 个 episode）")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--no-zero-shot", action="store_true",
                        help="跳过全场景 zero-shot 评估")
    args = parser.parse_args()

    target = None if args.target == "random" else args.target
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"❌ ckpt not found: {ckpt_path}")
        return

    print(f"ckpt:     {ckpt_path}")
    print(f"target:   {args.target}")
    print(f"episodes: {args.episodes} per regime")

    render_dir = REPO_ROOT / "eval_renders" / "final"
    render_dir.mkdir(parents=True, exist_ok=True)

    # ── 训练分布内（radius=0.10）──
    env_train = GraspEnv(
        target_object=target, max_episode_steps=args.max_steps,
        curriculum_radius=0.10, render_mode="rgb_array",
    )
    model = SAC.load(args.ckpt, env=env_train)
    result_train = run_eval(
        model, env_train, args.episodes,
        label="训练分布 (radius=0.10)",
        render_dir=render_dir, n_videos=args.n_videos,
    )
    env_train.close()

    # ── 全场景 zero-shot ──
    result_zs = None
    if not args.no_zero_shot:
        env_full = GraspEnv(
            target_object=target, max_episode_steps=args.max_steps,
            curriculum_radius=None, render_mode="rgb_array",
        )
        # 重新加载 model（绑定新 env）
        model_zs = SAC.load(args.ckpt, env=env_full)
        result_zs = run_eval(
            model_zs, env_full, args.episodes,
            label="zero-shot 全场景",
            render_dir=render_dir, n_videos=args.n_videos,
        )
        env_full.close()

    # ── 最终对比 ──
    print(f"\n{'='*60}")
    print(f"  最终对比")
    print(f"{'='*60}")
    print(f"{'指标':<14} {'训练分布':<12} {'zero-shot':<12}")
    if result_zs:
        for key in ["approached", "lifted", "held", "placed", "oob"]:
            t = result_train["stats"][key] / result_train["n"]
            z = result_zs["stats"][key] / result_zs["n"]
            print(f"  {key:<12} {t:>6.0%}      {z:>6.0%}")
    print(f"\n视频保存在: {render_dir}")


if __name__ == "__main__":
    main()

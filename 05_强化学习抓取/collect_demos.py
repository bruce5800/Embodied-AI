"""collect_demos.py — 跑 expert，过滤 placed=True trajectory，存 npz 供 BC 用。

每次 env.step 通过 TransitionRecorder 包装，记录 (obs, action, reward, next_obs, done)。
episode 结束后只保留 placed=True 的 trajectory。

支持多进程并行（默认 1，建议 4-6）。

用法：
    # 单进程
    python collect_demos.py --target blue_cube --n-success 50

    # 4 进程并行（M 系列 Mac 推荐）
    python collect_demos.py --target blue_cube --n-success 50 --n-workers 4

    # 后台跑长时间
    python collect_demos.py --target blue_cube --n-success 100 --n-workers 6 &
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent


class TransitionRecorder:
    """env wrapper：每次 step 记录 (prev_obs, action, reward, next_obs, done)。

    Python attribute lookup: 自定义方法 step/reset 在 wrapper class 中先解析；
    其他属性走 __getattr__ 转发给内部 env（_data, _model 等）。
    """

    def __init__(self, env):
        self.env = env
        self.transitions: list[dict] = []
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs.copy()
        self.transitions = []
        return obs, info

    def step(self, action):
        prev_obs = self._last_obs
        obs, r, term, trunc, info = self.env.step(action)
        self.transitions.append({
            "obs": prev_obs.copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "reward": float(r),
            "next_obs": obs.copy(),
            "done": bool(term or trunc),
        })
        self._last_obs = obs.copy()
        return obs, r, term, trunc, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def _collect_worker(args: dict) -> dict:
    """单 worker：跑 expert，只返回 placed=True 的 trajectories。

    必须是 module 顶层函数（spawn 启动需要 pickle-able）。
    """
    # 进程内部 import（spawn 模式下每个 worker 独立 import）
    from env import GraspEnv
    from expert import run_episode

    env = GraspEnv(
        target_object=args["target"],
        max_episode_steps=args["max_steps"],
        soften_contacts=True,    # expert 抓取必须软接触
    )
    rec_env = TransitionRecorder(env)

    collected: list[list[dict]] = []   # 每个元素是一条成功的 trajectory
    n_success = 0
    n_attempts = 0
    wid = args["worker_id"]

    for ep in range(args["max_attempts"]):
        n_attempts += 1
        result = run_episode(rec_env, seed=args["seed_base"] + ep)
        if result["placed"]:
            n_success += 1
            collected.append(rec_env.transitions[:])  # copy
            n_trans = len(rec_env.transitions)
            print(f"  [w{wid}] ep {ep:4d}: placed ✓  "
                  f"trans={n_trans:3d}  worker_progress {n_success}/{args['n_target']}",
                  flush=True)
            if n_success >= args["n_target"]:
                break

    env.close()
    # flatten 成单 list of transitions
    flat = [t for traj in collected for t in traj]
    return {
        "transitions": flat,
        "n_trajectories": len(collected),
        "n_attempts": n_attempts,
        "worker_id": wid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="blue_cube")
    parser.add_argument("--n-success", type=int, default=50,
                        help="目标 placed=True trajectory 数（多 worker 时按 worker 均分）")
    parser.add_argument("--max-attempts", type=int, default=1000,
                        help="单 worker 最大 attempts；总 attempts = n_workers × max_attempts")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="并行 worker 数（M 系列 Mac 推荐 4-6）")
    parser.add_argument("--out", default=None,
                        help="输出 npz 路径，默认 demos/{target}_v9.npz")
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--seed-base", type=int, default=50_000)
    args = parser.parse_args()

    out_path = (Path(args.out) if args.out
                else REPO_ROOT / "demos" / f"{args.target}_v9.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("─" * 60)
    print(f"  target          : {args.target}")
    print(f"  n_success goal  : {args.n_success}")
    print(f"  n_workers       : {args.n_workers}")
    print(f"  max_attempts/wk : {args.max_attempts}")
    print(f"  output          : {out_path}")
    print("─" * 60)

    n_workers = max(1, args.n_workers)

    # 每个 worker 平均分配 n_success target；多收集一些以防某些 worker placed 率低
    per_worker_target = (args.n_success + n_workers - 1) // n_workers

    worker_configs = [
        {
            "worker_id": i,
            "target": args.target,
            # seed 范围隔得远，避免不同 worker 跑同样 episode
            "seed_base": args.seed_base + i * 1_000_000,
            "max_attempts": args.max_attempts,
            "n_target": per_worker_target,
            "max_steps": args.max_episode_steps,
        }
        for i in range(n_workers)
    ]

    t0 = time.time()
    if n_workers == 1:
        results = [_collect_worker(worker_configs[0])]
    else:
        # macOS + MuJoCo: 必须 spawn（fork 会段错误）
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_collect_worker, worker_configs)
    elapsed = time.time() - t0

    # 合并
    all_obs, all_acts, all_rew, all_nobs, all_done = [], [], [], [], []
    total_attempts = 0
    total_trajectories = 0
    for r in results:
        total_attempts += r["n_attempts"]
        total_trajectories += r["n_trajectories"]
        for t in r["transitions"]:
            all_obs.append(t["obs"])
            all_acts.append(t["action"])
            all_rew.append(t["reward"])
            all_nobs.append(t["next_obs"])
            all_done.append(t["done"])

    print("─" * 60)
    print(f"  elapsed         : {elapsed/60:.1f} min")
    print(f"  total attempts  : {total_attempts}")
    print(f"  total trajectories: {total_trajectories}")
    print(f"  total transitions: {len(all_obs):,}")
    print(f"  placed rate     : {total_trajectories/total_attempts:.0%}")
    print("─" * 60)
    for r in results:
        print(f"  worker {r['worker_id']}: "
              f"{r['n_trajectories']} traj / {r['n_attempts']} attempts")

    if not all_obs:
        print("\n❌ 0 transitions, 不保存。")
        return

    np.savez(
        out_path,
        obs=np.asarray(all_obs, dtype=np.float32),
        action=np.asarray(all_acts, dtype=np.float32),
        reward=np.asarray(all_rew, dtype=np.float32),
        next_obs=np.asarray(all_nobs, dtype=np.float32),
        done=np.asarray(all_done, dtype=np.bool_),
    )
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"\n✓ 保存 {len(all_obs):,} transitions ({total_trajectories} trajectories) "
          f"→ {out_path}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()

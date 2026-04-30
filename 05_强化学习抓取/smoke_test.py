"""GraspEnv 冒烟测试 — 50 步随机动作，验证 reset/step/spaces/info 都通。

跑：
    cd 05_强化学习抓取
    python smoke_test.py
"""

import numpy as np

from env import GraspEnv


def main():
    env = GraspEnv()
    obs, info = env.reset(seed=42)

    print("─" * 60)
    print(f"observation_space  : {env.observation_space}")
    print(f"action_space       : {env.action_space}")
    print(f"obs.shape          : {obs.shape}  dtype={obs.dtype}")
    print(f"target_object      : {info['target_object']}")
    print(f"target_zone        : {info['target_zone']}")
    print(f"initial obj_pos    : {info['object_pos']}")
    print(f"initial ee_pos     : {info['ee_pos']}")
    print("─" * 60)

    total_reward = 0.0
    n_steps = 50
    for t in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"episode ended at step {t+1}: "
                  f"terminated={terminated} truncated={truncated}")
            break

    print(f"steps run          : {info['steps']}")
    print(f"total reward       : {total_reward:.3f}")
    print(f"avg reward / step  : {total_reward / max(info['steps'], 1):.3f}")
    print(f"final obj_pos      : {info['object_pos']}")
    print(f"final ee_pos       : {info['ee_pos']}")
    print(f"object_lifted      : {info['object_lifted']}")

    # 形状/类型断言
    assert obs.shape == (28,), f"obs shape: {obs.shape}"
    assert obs.dtype == np.float32
    assert env.action_space.shape == (6,)
    print("─" * 60)
    print("OK")

    env.close()


if __name__ == "__main__":
    main()

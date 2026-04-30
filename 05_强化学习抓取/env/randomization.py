"""物体位姿随机化 — 从 04_/vision_pipeline.py:_randomize_objects() 拆出。

每个 episode reset 时调一次，给 4 个 freejoint 物体随机 (x, y) + 随机 yaw。
"""

import numpy as np
import mujoco

from .scene_constants import OBJECTS, RANDOM_X_RANGE, RANDOM_Y_RANGE


def randomize_objects(model, data, np_random=None):
    """随机化所有目标物体的位置和朝向。

    Args:
        model: MjModel
        data: MjData（原地修改 qpos / qvel）
        np_random: numpy Generator；为 None 则用全局 RNG
    """
    if np_random is None:
        np_random = np.random.default_rng()

    for obj in OBJECTS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj["name"])
        if body_id < 0:
            continue

        jnt_id = model.body_jntadr[body_id]
        if jnt_id < 0:
            continue

        qpos_adr = model.jnt_qposadr[jnt_id]
        qvel_adr = model.jnt_dofadr[jnt_id]

        # 随机 (x, y)，z 略高于地面让物体自然下落
        x = float(np_random.uniform(*RANDOM_X_RANGE))
        y = float(np_random.uniform(*RANDOM_Y_RANGE))
        z = 0.05

        data.qpos[qpos_adr + 0] = x
        data.qpos[qpos_adr + 1] = y
        data.qpos[qpos_adr + 2] = z

        # 随机绕 Z 轴旋转
        angle = float(np_random.uniform(0.0, 2.0 * np.pi))
        data.qpos[qpos_adr + 3] = np.cos(angle / 2.0)
        data.qpos[qpos_adr + 4] = 0.0
        data.qpos[qpos_adr + 5] = 0.0
        data.qpos[qpos_adr + 6] = np.sin(angle / 2.0)

        # 清零速度
        data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    mujoco.mj_forward(model, data)

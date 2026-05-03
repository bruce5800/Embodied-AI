"""物体位姿随机化 — 从 04_/vision_pipeline.py:_randomize_objects() 拆出。

每个 episode reset 时调一次，给 4 个 freejoint 物体随机 (x, y) + 随机 yaw。

reverse curriculum 支持：可指定 target_override，把单个 target 物体放到指定中心
±radius 内，绕过全场景随机化。
"""

import numpy as np
import mujoco

from .scene_constants import OBJECTS, RANDOM_X_RANGE, RANDOM_Y_RANGE


def _set_object_pose(model, data, body_name, x, y, z, yaw):
    """把单个 freejoint 物体设到 (x, y, z) + 绕 z 轴 yaw 弧度。"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return
    jnt_id = model.body_jntadr[body_id]
    if jnt_id < 0:
        return

    qpos_adr = model.jnt_qposadr[jnt_id]
    qvel_adr = model.jnt_dofadr[jnt_id]

    data.qpos[qpos_adr + 0] = float(x)
    data.qpos[qpos_adr + 1] = float(y)
    data.qpos[qpos_adr + 2] = float(z)
    data.qpos[qpos_adr + 3] = float(np.cos(yaw / 2.0))
    data.qpos[qpos_adr + 4] = 0.0
    data.qpos[qpos_adr + 5] = 0.0
    data.qpos[qpos_adr + 6] = float(np.sin(yaw / 2.0))
    data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def randomize_objects(
    model,
    data,
    np_random=None,
    target_override=None,
):
    """随机化所有物体的位置和朝向。

    Args:
        model:           MjModel
        data:            MjData（原地修改 qpos / qvel）
        np_random:       numpy Generator；None 则用全局 RNG
        target_override: dict 或 None，None 表示走默认全场景随机。
                         dict 形如 {
                             "name": "red_cylinder",
                             "center": np.array([-0.15, -0.35]),  # xy
                             "radius": 0.05,                      # 方形半边长
                         }
                         指定后，target 物体在 center ± radius 方框内随机；
                         其他物体仍按全场景随机（远离 target）。
    """
    if np_random is None:
        np_random = np.random.default_rng()

    target_name = target_override["name"] if target_override else None

    for obj in OBJECTS:
        if obj["name"] == target_name:
            continue  # target 后面单独处理
        # 随机 (x, y)，z 略高于地面让物体自然下落
        x = float(np_random.uniform(*RANDOM_X_RANGE))
        y = float(np_random.uniform(*RANDOM_Y_RANGE))
        yaw = float(np_random.uniform(0.0, 2.0 * np.pi))
        _set_object_pose(model, data, obj["name"], x, y, 0.05, yaw)

    if target_override is not None:
        cx, cy = target_override["center"][:2]
        r = float(target_override["radius"])
        # 课程化范围 clip 到全场景 RANDOM_X/Y_RANGE 内，避免 obj 跑到工作空间外
        x = float(np.clip(cx + np_random.uniform(-r, r),
                          RANDOM_X_RANGE[0], RANDOM_X_RANGE[1]))
        y = float(np.clip(cy + np_random.uniform(-r, r),
                          RANDOM_Y_RANGE[0], RANDOM_Y_RANGE[1]))
        yaw = float(np_random.uniform(0.0, 2.0 * np.pi))
        _set_object_pose(model, data, target_override["name"], x, y, 0.05, yaw)

    mujoco.mj_forward(model, data)

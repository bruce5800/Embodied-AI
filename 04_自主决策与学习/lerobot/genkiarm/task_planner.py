"""
任务规划 — 放置指令解析 + 默认分拣规则
将结构化指令（zone/absolute/relative/line）解析为具体坐标。
"""

import mujoco
import numpy as np

from config import ZONES, DEFAULT_SORT_RULES


def resolve_place_target(instruction, mj_model, mj_data):
    """
    将一条放置指令解析为具体的 (object_name, [x, y, z]) 列表。

    支持 4 种模式:
      zone     — 放到预定义区域
      absolute — 放到绝对坐标
      relative — 放到另一物体的相对位置
      line     — 多个物体排成一排

    Returns:
        [(object_name, target_xyz), ...]
    """
    mujoco.mj_forward(mj_model, mj_data)
    mode = instruction.get("mode", "zone")

    if mode == "zone":
        obj = instruction["object"]
        zone_name = instruction.get("zone", DEFAULT_SORT_RULES.get(obj))
        if zone_name is None or zone_name not in ZONES:
            print(f"  ✗ 未知放置区: {zone_name}")
            return []
        return [(obj, ZONES[zone_name].copy())]

    elif mode == "absolute":
        obj = instruction["object"]
        pos = np.array(instruction["position"], dtype=np.float64)
        return [(obj, pos)]

    elif mode == "relative":
        obj = instruction["object"]
        ref = instruction["ref_object"]
        relation = instruction.get("relation", "above")
        offset_val = instruction.get("offset", 0.08)

        ref_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, ref)
        if ref_body_id < 0:
            print(f"  ✗ 参考物体 {ref} 不存在")
            return []
        ref_pos = mj_data.xpos[ref_body_id].copy()

        offset_map = {
            "above":  np.array([0.0, 0.0, offset_val]),
            "below":  np.array([0.0, 0.0, -offset_val]),
            "left":   np.array([0.0, offset_val, 0.0]),
            "right":  np.array([0.0, -offset_val, 0.0]),
            "front":  np.array([offset_val, 0.0, 0.0]),
            "behind": np.array([-offset_val, 0.0, 0.0]),
        }
        offset = offset_map.get(relation, np.array([0.0, 0.0, offset_val]))
        target = ref_pos + offset
        return [(obj, target)]

    elif mode == "line":
        objects = instruction["objects"]
        start = np.array(instruction["start"], dtype=np.float64)
        end = np.array(instruction["end"], dtype=np.float64)
        n = len(objects)
        tasks = []
        for i, obj in enumerate(objects):
            t = i / max(n - 1, 1)
            pos = start + t * (end - start)
            tasks.append((obj, pos))
        return tasks

    else:
        print(f"  ✗ 未知放置模式: {mode}")
        return []


def build_default_instructions(detections):
    """没有外部指令时，使用默认分拣规则生成指令列表。"""
    instructions = []
    for _name, _conf, body_name, _pos in detections:
        if body_name in DEFAULT_SORT_RULES:
            instructions.append({
                "object": body_name,
                "mode": "zone",
                "zone": DEFAULT_SORT_RULES[body_name],
            })
    return instructions

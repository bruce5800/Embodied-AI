"""
运动控制 — 渲染、移动、YOLO 检测、抓取/放置动作
"""

import cv2
import mujoco
import numpy as np

from config import (
    N_ARM_JOINTS, GRIPPER_IDX, GRIPPER_OPEN, GRIPPER_CLOSE,
    PRE_GRASP_HEIGHT, LIFT_HEIGHT, GRASP_OFFSET,
    PLACE_HEIGHT, PLACE_HOVER_HEIGHT,
    GROUND_Z_THRESHOLD, MAX_GRASP_ATTEMPTS,
    READY_JOINTS, OBJECTS, COLORS, WINDOW,
)
from ik_solver import solve_ik


# ═══════════════════════════════════════════════════════════════
#  渲染 & 可视化
# ═══════════════════════════════════════════════════════════════

def render_frame(model, data, renderer, cam_name, stage="", extra_text=""):
    """渲染一帧并在画面上叠加状态信息"""
    renderer.update_scene(data, camera=cam_name)
    img = renderer.render()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if stage:
        cv2.putText(img_bgr, stage, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if extra_text:
        cv2.putText(img_bgr, extra_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(WINDOW, img_bgr)
    cv2.waitKey(1)
    return img_bgr


def move_to(model, data, target_angles, renderer, cam_name,
            stage="Moving", steps=400):
    """
    平滑移动臂体到目标关节角度。
    通过 ctrl 设置目标位置，MuJoCo PD 控制器自然过渡。
    """
    data.ctrl[:N_ARM_JOINTS] = target_angles

    for step in range(steps):
        mujoco.mj_step(model, data)

        if step % 5 == 0:
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
            ee_pos = data.site_xpos[site_id]
            info = f"EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
            render_frame(model, data, renderer, cam_name, stage, info)


# ═══════════════════════════════════════════════════════════════
#  YOLO 检测
# ═══════════════════════════════════════════════════════════════

def detect_objects(yolo, model, data, renderer, cam_name="camera_top"):
    """
    YOLO 检测 → 匹配 MuJoCo body → 获取 3D 位置。

    Returns:
        detections: [(class_name, confidence, body_name, pos_3d), ...]
        det_img: 标注了检测框的图像
    """
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam_name)
    img = renderer.render()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = yolo.predict(img_bgr, verbose=False, conf=0.5)

    detections = []
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            if cls_id < len(OBJECTS):
                obj = OBJECTS[cls_id]
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, obj["name"])
                body_pos = data.xpos[body_id].copy()
                detections.append(
                    (obj["class_name"], conf, obj["name"], body_pos))

                color = COLORS[cls_id] if cls_id < len(COLORS) else (255, 255, 255)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                label = f"{obj['class_name']} {conf:.2f}"
                cv2.putText(img_bgr, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return detections, img_bgr


# ═══════════════════════════════════════════════════════════════
#  抓取 / 放置 / 验证
# ═══════════════════════════════════════════════════════════════

def execute_grasp(model, data, target_pos, renderer, cam_name="camera_front"):
    """
    抓取序列：就绪 → 预抓取悬停 → 下降 → 闭合夹爪 → 抬起
    """
    # 阶段 1: 就绪
    print("  [1/5] 就绪姿态 + 张开夹爪")
    data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
    move_to(model, data, READY_JOINTS, renderer, cam_name,
            stage="[1/5] Ready", steps=300)

    # 阶段 2: 预抓取悬停
    grasp_pos = target_pos + GRASP_OFFSET
    pre_grasp = grasp_pos.copy()
    pre_grasp[2] += PRE_GRASP_HEIGHT

    print(f"  [2/5] 预抓取悬停 → ({pre_grasp[0]:.3f}, {pre_grasp[1]:.3f}, {pre_grasp[2]:.3f})")
    angles, ok = solve_ik(model, pre_grasp, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（预抓取位置不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[2/5] Pre-grasp", steps=500)

    # 阶段 3: 下降
    print(f"  [3/5] 下降 → ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
    angles, ok = solve_ik(model, grasp_pos, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（抓取位置不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[3/5] Descend", steps=400)

    # 阶段 4: 闭合夹爪
    print("  [4/5] 闭合夹爪")
    data.ctrl[GRIPPER_IDX] = GRIPPER_CLOSE
    move_to(model, data, data.ctrl[:N_ARM_JOINTS].copy(), renderer, cam_name,
            stage="[4/5] Grasping", steps=300)

    # 阶段 5: 抬起
    lift_pos = grasp_pos.copy()
    lift_pos[2] += LIFT_HEIGHT

    print(f"  [5/5] 抬起 → ({lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f})")
    angles, ok = solve_ik(model, lift_pos, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（抬起位置不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[5/5] Lifting", steps=500)

    render_frame(model, data, renderer, cam_name, stage="Done!")
    return True


def check_grasp_success(model, data, body_name, renderer, cam_name):
    """验证：物体 z 坐标高于阈值 = 被抬起来了"""
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    obj_z = data.xpos[body_id][2]
    lifted = obj_z > GROUND_Z_THRESHOLD

    status = f"物体高度: {obj_z:.3f}m — {'已抬起' if lifted else '仍在地面'}"
    print(f"  验证: {status}")
    render_frame(model, data, renderer, cam_name,
                 stage="OK" if lifted else "MISS", extra_text=status)
    return lifted


def execute_place(model, data, place_pos, renderer, cam_name="camera_front"):
    """
    放置序列：悬停 → 下降 → 松手 → 抬起 → 就绪
    """
    zone_pos = np.array(place_pos, dtype=np.float64)
    print(f"\n  放置目标 → ({zone_pos[0]:.3f}, {zone_pos[1]:.3f}, {zone_pos[2]:.3f})")

    # 阶段 1: 悬停
    hover_pos = zone_pos.copy()
    hover_pos[2] += PLACE_HOVER_HEIGHT
    print(f"  [放置 1/5] 悬停 → ({hover_pos[0]:.3f}, {hover_pos[1]:.3f}, {hover_pos[2]:.3f})")
    angles, ok = solve_ik(model, hover_pos, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（放置区悬停不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[Place 1/5] Hover", steps=500)

    # 阶段 2: 下降
    place_target = zone_pos.copy()
    place_target[2] += PLACE_HEIGHT
    print(f"  [放置 2/5] 下降 → ({place_target[0]:.3f}, {place_target[1]:.3f}, {place_target[2]:.3f})")
    angles, ok = solve_ik(model, place_target, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（放置位置不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[Place 2/5] Descend", steps=400)

    # 阶段 3: 松手
    print("  [放置 3/5] 松开夹爪")
    data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
    move_to(model, data, data.ctrl[:N_ARM_JOINTS].copy(), renderer, cam_name,
            stage="[Place 3/5] Release", steps=300)

    # 阶段 4: 抬起
    retreat_pos = place_target.copy()
    retreat_pos[2] += PLACE_HOVER_HEIGHT
    print(f"  [放置 4/5] 抬起 → ({retreat_pos[0]:.3f}, {retreat_pos[1]:.3f}, {retreat_pos[2]:.3f})")
    angles, ok = solve_ik(model, retreat_pos, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（抬起不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[Place 4/5] Retreat", steps=400)

    # 阶段 5: 回到就绪位
    print("  [放置 5/5] 回到就绪位")
    move_to(model, data, READY_JOINTS, renderer, cam_name,
            stage="[Place 5/5] Ready", steps=300)

    render_frame(model, data, renderer, cam_name, stage="Placed!")
    return True


def return_to_ready(model, data, renderer, cam_name):
    """回到就绪位置并张开夹爪。"""
    data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
    move_to(model, data, READY_JOINTS, renderer, cam_name,
            stage="Retry - Ready", steps=400)
    for _ in range(500):
        mujoco.mj_step(model, data)

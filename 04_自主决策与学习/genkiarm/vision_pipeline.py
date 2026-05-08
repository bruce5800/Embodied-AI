"""
MuJoCo + YOLO 视觉感知 Pipeline
================================
两个功能：
  1. generate  — 从 MuJoCo 仿真自动生成 YOLO 训练数据集
                  随机化物体位置/姿态 → 多相机渲染 → 自动计算 BBox 标注
  2. detect    — 实时检测：MuJoCo 渲染 → YOLO 推理 → 标注框 + 类别

用法：
  # 1) 生成训练数据（默认 500 张）
  python vision_pipeline.py generate --num 500

  # 2) 训练 YOLO（用 ultralytics CLI）
  yolo task=detect mode=train model=yolov8n.pt data=dataset/dataset.yaml epochs=80 imgsz=640

  # 3) 实时检测
  python vision_pipeline.py detect --model runs/detect/train/weights/best.pt
"""

import argparse
import os
import random
import time

import cv2
import mujoco
import numpy as np

# ─── 路径 ───
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(_SCRIPT_DIR, "asserts", "lift_cube2.xml")

# ─── 场景中的目标物体定义 ───
# name: MuJoCo body 名称
# class_id: YOLO 类别 ID
# class_name: 类别标签
OBJECTS = [
    {"name": "red_cylinder",    "class_id": 0, "class_name": "red_cylinder"},
    {"name": "blue_cube",       "class_id": 1, "class_name": "blue_cube"},
    {"name": "green_sphere",    "class_id": 2, "class_name": "green_sphere"},
    {"name": "yellow_cylinder", "class_id": 3, "class_name": "yellow_cylinder"},
]

# 相机列表
CAMERAS = ["camera_front", "camera_top", "camera_side"]

# 物体随机化范围（仿真桌面区域）
RANDOM_X_RANGE = (-0.45, -0.15)
RANDOM_Y_RANGE = (-0.25, 0.25)


def _project_to_pixel(model, data, renderer, camera_name, point_3d, img_w, img_h):
    """
    将 MuJoCo 世界坐标投影到相机像素坐标。
    返回 (u, v) 像素坐标，如果在画面外返回 None。
    """
    # 获取相机 ID
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        return None

    # 相机位姿
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    # 世界坐标 → 相机坐标
    delta = point_3d - cam_pos
    cam_coord = cam_mat.T @ delta  # 相机坐标系: x右, y下, z前（OpenGL 惯例取反）

    # MuJoCo 相机: x右, y上, z后（OpenGL 惯例）
    # 所以需要翻转
    x_cam = cam_coord[0]
    y_cam = -cam_coord[1]
    z_cam = -cam_coord[2]

    if z_cam <= 0:
        return None  # 在相机后方

    # 相机内参（MuJoCo 用 fovy）
    fovy = model.cam_fovy[cam_id]
    fovy_rad = np.radians(fovy)
    f_y = img_h / (2.0 * np.tan(fovy_rad / 2.0))
    f_x = f_y  # 假设 aspect ratio = 1 的焦距，再按实际 aspect 调整
    # MuJoCo renderer 的 aspect = img_w / img_h
    # f_x = f_y 对于正方形像素

    u = img_w / 2.0 + f_x * (x_cam / z_cam)
    v = img_h / 2.0 + f_y * (y_cam / z_cam)

    return (u, v)


def _get_bbox_for_object(model, data, renderer, camera_name, obj_body_name,
                         img_w, img_h, margin=15):
    """
    计算物体在相机图像中的 2D BBox。
    通过投影物体中心 ± 几何尺寸的多个点来估计 BBox。
    返回 (x_center, y_center, w, h) 归一化到 [0,1]，或 None。
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name)
    if body_id < 0:
        return None

    # 物体中心位置
    body_pos = data.xpos[body_id].copy()

    # 获取该 body 的所有 geom，找最大尺寸
    max_size = 0.04  # 默认半径
    for gi in range(model.nbody):
        pass
    # 简化：用固定的包围半径（物体都很小）
    radius = 0.04

    # 生成包围球上的采样点
    offsets = [
        [radius, 0, 0], [-radius, 0, 0],
        [0, radius, 0], [0, -radius, 0],
        [0, 0, radius], [0, 0, -radius],
        [0, 0, 0],  # 中心
    ]

    pixels = []
    for off in offsets:
        pt = body_pos + np.array(off)
        pix = _project_to_pixel(model, data, renderer, camera_name, pt, img_w, img_h)
        if pix is not None:
            pixels.append(pix)

    if len(pixels) < 2:
        return None

    us = [p[0] for p in pixels]
    vs = [p[1] for p in pixels]

    u_min = max(0, min(us) - margin)
    u_max = min(img_w, max(us) + margin)
    v_min = max(0, min(vs) - margin)
    v_max = min(img_h, max(vs) + margin)

    # 太小或完全出画
    box_w = u_max - u_min
    box_h = v_max - v_min
    if box_w < 5 or box_h < 5:
        return None

    # 归一化
    cx = (u_min + u_max) / 2.0 / img_w
    cy = (v_min + v_max) / 2.0 / img_h
    nw = box_w / img_w
    nh = box_h / img_h

    # 确保在 [0, 1] 范围内
    cx = np.clip(cx, 0, 1)
    cy = np.clip(cy, 0, 1)
    nw = np.clip(nw, 0, 1)
    nh = np.clip(nh, 0, 1)

    return (cx, cy, nw, nh)


def _randomize_objects(model, data):
    """随机化所有目标物体的位置（在桌面范围内）"""
    for obj in OBJECTS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj["name"])
        if body_id < 0:
            continue

        # 获取该 body 对应的 freejoint 的 qpos 索引
        jnt_id = model.body_jntadr[body_id]
        if jnt_id < 0:
            continue

        qpos_adr = model.jnt_qposadr[jnt_id]

        # 随机位置
        x = random.uniform(*RANDOM_X_RANGE)
        y = random.uniform(*RANDOM_Y_RANGE)
        z = 0.05  # 略高于地面让物体自然落下

        data.qpos[qpos_adr + 0] = x
        data.qpos[qpos_adr + 1] = y
        data.qpos[qpos_adr + 2] = z

        # 随机朝向（四元数）
        angle = random.uniform(0, 2 * np.pi)
        data.qpos[qpos_adr + 3] = np.cos(angle / 2)
        data.qpos[qpos_adr + 4] = 0
        data.qpos[qpos_adr + 5] = 0
        data.qpos[qpos_adr + 6] = np.sin(angle / 2)

        # 清零速度
        qvel_adr = model.jnt_dofadr[jnt_id]
        data.qvel[qvel_adr:qvel_adr + 6] = 0

    mujoco.mj_forward(model, data)


def generate_dataset(num_samples=500, img_size=640, output_dir=None):
    """
    从 MuJoCo 仿真自动生成 YOLO 训练数据。
    每次随机化物体位置 → 物理稳定 → 多相机渲染 → 自动标注。
    """
    if output_dir is None:
        output_dir = os.path.join(_SCRIPT_DIR, "dataset")

    # 创建目录结构
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # 写 dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    class_names = [obj["class_name"] for obj in OBJECTS]
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(OBJECTS)}\n")
        f.write(f"names: {class_names}\n")

    print(f"数据集目录: {output_dir}")
    print(f"类别: {class_names}")
    print(f"生成 {num_samples} 组场景 × {len(CAMERAS)} 相机 = {num_samples * len(CAMERAS)} 张图片")
    print()

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=img_size, width=img_size)

    # 机械臂收起（不遮挡物体）
    data.ctrl[:5] = [0, -1.5, 1.0, 0.5, 0]  # 收到后方
    data.ctrl[5] = np.radians(90)  # 夹爪张开

    val_ratio = 0.2  # 20% 作为验证集
    total = 0
    empty = 0

    for i in range(num_samples):
        # 随机化物体位置
        _randomize_objects(model, data)

        # 物理稳定（让物体落到桌面上）
        for _ in range(500):
            mujoco.mj_step(model, data)

        # 再次前向计算确保位姿准确
        mujoco.mj_forward(model, data)

        # 选择 train 或 val
        split = "val" if random.random() < val_ratio else "train"

        for cam_idx, cam_name in enumerate(CAMERAS):
            img_id = f"{i:05d}_{cam_name}"

            # 渲染
            renderer.update_scene(data, camera=cam_name)
            img = renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 计算标注
            labels = []
            for obj in OBJECTS:
                bbox = _get_bbox_for_object(
                    model, data, renderer, cam_name,
                    obj["name"], img_size, img_size
                )
                if bbox is not None:
                    cx, cy, w, h = bbox
                    labels.append(f"{obj['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # 保存图片和标注
            img_path = os.path.join(output_dir, "images", split, f"{img_id}.jpg")
            lbl_path = os.path.join(output_dir, "labels", split, f"{img_id}.txt")

            cv2.imwrite(img_path, img_bgr)
            with open(lbl_path, "w") as f:
                f.write("\n".join(labels))

            total += 1
            if len(labels) == 0:
                empty += 1

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{num_samples} ({total} 张图片, {empty} 张无标注)")

    renderer.close()
    print(f"\n完成! 共 {total} 张图片")
    print(f"  有标注: {total - empty}  无标注: {empty}")
    print(f"  dataset.yaml: {yaml_path}")
    print(f"\n下一步训练:")
    print(f"  yolo task=detect mode=train model=yolov8n.pt data={yaml_path} epochs=80 imgsz={img_size}")


def run_detection(model_path=None, camera="camera_top"):
    """
    实时检测 pipeline：MuJoCo 渲染 → YOLO 推理 → 可视化。
    同时支持键盘控制机械臂（与 mujoco_demo.py 的 cv 模式一致）。
    """
    from ultralytics import YOLO

    if model_path is None:
        # 尝试找默认训练好的模型
        default_paths = [
            os.path.join(_SCRIPT_DIR, "runs", "detect", "train", "weights", "best.pt"),
            os.path.join(_SCRIPT_DIR, "best.pt"),
        ]
        for p in default_paths:
            if os.path.exists(p):
                model_path = p
                break
        if model_path is None:
            print("错误: 未找到 YOLO 模型，请指定 --model 路径")
            print("  或先运行: python vision_pipeline.py generate")
            print("  然后训练: yolo task=detect mode=train model=yolov8n.pt data=dataset/dataset.yaml epochs=80")
            return

    print(f"YOLO 模型: {model_path}")
    print(f"相机: {camera}")

    # 加载 YOLO
    yolo = YOLO(model_path)
    class_names = [obj["class_name"] for obj in OBJECTS]
    colors = [
        (0, 0, 255),    # 红
        (255, 100, 0),  # 蓝
        (0, 200, 0),    # 绿
        (0, 220, 255),  # 黄
    ]

    # 加载 MuJoCo
    mj_model = mujoco.MjModel.from_xml_path(SCENE_XML)
    mj_data = mujoco.MjData(mj_model)

    width, height = 640, 480
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # 控制状态
    arm_targets = np.zeros(mj_model.nu)
    arm_targets[5] = np.radians(90)  # 夹爪张开
    selected_joint = 0
    joint_step = np.radians(5)
    gripper_open = True
    cam_idx = CAMERAS.index(camera) if camera in CAMERAS else 0

    joint_names = ['底座', '肩部', '肘部', '腕俯仰', '腕旋转']

    print("操控: 1~5选关节 ←→控制 ↑↓步长 Space夹爪 0复位 9切相机 ESC退出")

    while True:
        # 仿真
        mj_data.ctrl[:] = arm_targets
        for _ in range(10):
            mujoco.mj_step(mj_model, mj_data)

        # 渲染
        renderer.update_scene(mj_data, camera=CAMERAS[cam_idx])
        img = renderer.render()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # YOLO 推理
        results = yolo.predict(img_bgr, verbose=False, conf=0.5)

        # 绘制检测结果
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                color = colors[cls_id] if cls_id < len(colors) else (255, 255, 255)
                label = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"

                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_bgr, f"{label} {conf:.2f}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # HUD
        cv2.putText(img_bgr, f"Cam: {CAMERAS[cam_idx]}  J{selected_joint+1}-{joint_names[selected_joint]}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        grip_str = "OPEN" if gripper_open else "CLOSED"
        cv2.putText(img_bgr, f"Gripper: {grip_str}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if gripper_open else (0, 0, 255), 1)

        cv2.imshow("MuJoCo + YOLO Detection", img_bgr)

        # 键盘
        key = cv2.waitKey(16) & 0xFFFF
        if key == 27:
            break
        elif ord('1') <= key <= ord('5'):
            selected_joint = key - ord('1')
        elif key in (83, 3, 65363):
            arm_targets[selected_joint] += joint_step
        elif key in (81, 2, 65361):
            arm_targets[selected_joint] -= joint_step
        elif key in (82, 0, 65362):
            joint_step = min(joint_step * 2, np.radians(20))
        elif key in (84, 1, 65364):
            joint_step = max(joint_step / 2, np.radians(1))
        elif key == ord(' '):
            gripper_open = not gripper_open
            arm_targets[5] = np.radians(90) if gripper_open else np.radians(-5)
        elif key == ord('0'):
            arm_targets[:5] = 0
            arm_targets[5] = np.radians(90)
            gripper_open = True
            mujoco.mj_resetData(mj_model, mj_data)
        elif key == ord('9'):
            cam_idx = (cam_idx + 1) % len(CAMERAS)

    cv2.destroyAllWindows()
    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo + YOLO 视觉 Pipeline")
    sub = parser.add_subparsers(dest="command")

    # generate 子命令
    gen = sub.add_parser("generate", help="生成 YOLO 训练数据集")
    gen.add_argument("--num", type=int, default=500, help="场景数量（每场景 × 3相机）")
    gen.add_argument("--size", type=int, default=640, help="图片尺寸")
    gen.add_argument("--output", type=str, default=None, help="输出目录")

    # detect 子命令
    det = sub.add_parser("detect", help="实时 YOLO 检测")
    det.add_argument("--model", type=str, default=None, help="YOLO 权重路径")
    det.add_argument("--camera", type=str, default="camera_top", help="初始相机")

    args = parser.parse_args()

    if args.command == "generate":
        generate_dataset(num_samples=args.num, img_size=args.size, output_dir=args.output)
    elif args.command == "detect":
        run_detection(model_path=args.model, camera=args.camera)
    else:
        parser.print_help()

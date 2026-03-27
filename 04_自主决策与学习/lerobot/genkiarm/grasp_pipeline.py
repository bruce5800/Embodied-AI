"""
MuJoCo 抓取控制 Pipeline
========================
YOLO 检测 → 目标定位 → IK 求解 → 分阶段抓取

流程：
  1. 渲染场景 → YOLO 检测目标物体
  2. 获取目标 3D 位置（仿真中直接读取 body position）
  3. IK 求解：Jacobian 迭代法（使用 MuJoCo 内置 mj_jacSite）
  4. 分阶段执行：就绪 → 预抓取悬停 → 下降 → 闭合夹爪 → 抬起

用法：
  python grasp_pipeline.py --model runs/detect/train/weights/best.pt
  python grasp_pipeline.py --model best.pt --target blue_cube
"""

import argparse
import os

import cv2
import mujoco
import numpy as np

# ─── 路径 ───
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(_SCRIPT_DIR, "asserts", "lift_cube2.xml")

# ─── 物体定义（与 vision_pipeline.py 一致）───
OBJECTS = [
    {"name": "red_cylinder",    "class_id": 0, "class_name": "red_cylinder"},
    {"name": "blue_cube",       "class_id": 1, "class_name": "blue_cube"},
    {"name": "green_sphere",    "class_id": 2, "class_name": "green_sphere"},
    {"name": "yellow_cylinder", "class_id": 3, "class_name": "yellow_cylinder"},
]
CLASS_NAMES = [obj["class_name"] for obj in OBJECTS]
COLORS = [
    (0, 0, 255),    # 红
    (255, 100, 0),  # 蓝
    (0, 200, 0),    # 绿
    (0, 220, 255),  # 黄
]

# ─── 抓取参数 ───
N_ARM_JOINTS = 5                     # joint1-5 控制臂体
GRIPPER_IDX = 5                      # joint6 控制夹爪
GRIPPER_OPEN = np.radians(90)        # 夹爪张开
GRIPPER_CLOSE = np.radians(-5)       # 夹爪闭合
PRE_GRASP_HEIGHT = 0.08              # 预抓取悬停高度（物体上方 8cm）
GRASP_HEIGHT_OFFSET = 0.005          # 抓取时末端下探量
LIFT_HEIGHT = 0.15                   # 抬起高度
READY_JOINTS = [0.0, -0.8, 0.5, 0.3, 0.0]  # 就绪姿态


# ═══════════════════════════════════════════════════════════════
#  IK 求解器 — 基于 MuJoCo Jacobian + 阻尼最小二乘
# ═══════════════════════════════════════════════════════════════

def solve_ik(model, target_pos, current_qpos=None, max_iter=500, tol=1e-3):
    """
    使用 MuJoCo 内置 mj_jacSite 求解逆运动学。
    仅控制 joint1-5（臂体），不动 joint6（夹爪）。

    原理:
      1. 计算末端当前位置与目标的误差 e
      2. 通过 mj_jacSite 计算末端对关节的 Jacobian J
      3. 阻尼最小二乘: Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
      4. 更新关节角，限幅，重复直到收敛

    Returns:
        (joint_angles, success)
    """
    # 在临时 data 上求解，不影响仿真状态
    data_ik = mujoco.MjData(model)
    if current_qpos is not None:
        data_ik.qpos[:] = current_qpos[:]

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    target = np.array(target_pos, dtype=np.float64)

    for _ in range(max_iter):
        mujoco.mj_forward(model, data_ik)

        err = target - data_ik.site_xpos[site_id]
        dist = np.linalg.norm(err)

        if dist < tol:
            return data_ik.qpos[:N_ARM_JOINTS].copy(), True

        # Jacobian: 3×nv 矩阵
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data_ik, jacp, None, site_id)
        J = jacp[:, :N_ARM_JOINTS]

        # 阻尼最小二乘
        lam = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err)

        # 限制单步幅度，防止震荡
        max_step = 0.15
        scale = np.max(np.abs(dq))
        if scale > max_step:
            dq *= max_step / scale

        data_ik.qpos[:N_ARM_JOINTS] += dq

        # 关节限幅
        for j in range(N_ARM_JOINTS):
            lo, hi = model.jnt_range[j]
            data_ik.qpos[j] = np.clip(data_ik.qpos[j], lo, hi)

    # 最终精度检查
    mujoco.mj_forward(model, data_ik)
    final_err = np.linalg.norm(target - data_ik.site_xpos[site_id])
    return data_ik.qpos[:N_ARM_JOINTS].copy(), final_err < tol * 5


# ═══════════════════════════════════════════════════════════════
#  渲染 & 可视化
# ═══════════════════════════════════════════════════════════════

WINDOW = "Grasp Pipeline"


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
    每帧渲染并更新窗口。
    """
    data.ctrl[:N_ARM_JOINTS] = target_angles

    for step in range(steps):
        mujoco.mj_step(model, data)

        # 每 5 步渲染一帧（~60fps 效果）
        if step % 5 == 0:
            # 计算当前末端位置
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

    在仿真中我们可以直接用 data.xpos 拿到物体真实位置，
    YOLO 的作用是确认"看到了哪些物体"（模拟真实感知）。

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

                # 画检测框
                color = COLORS[cls_id] if cls_id < len(COLORS) else (255, 255, 255)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                label = f"{obj['class_name']} {conf:.2f}"
                cv2.putText(img_bgr, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return detections, img_bgr


# ═══════════════════════════════════════════════════════════════
#  抓取序列
# ═══════════════════════════════════════════════════════════════

def execute_grasp(model, data, target_pos, renderer, cam_name="camera_front"):
    """
    执行完整抓取序列：
      阶段 1 — 就绪姿态 + 张开夹爪
      阶段 2 — 移动到预抓取位置（目标上方）
      阶段 3 — 下降到抓取位置
      阶段 4 — 闭合夹爪
      阶段 5 — 抬起

    Returns:
        success: bool
    """
    # ── 阶段 1: 就绪姿态 ──
    print("  [1/5] 就绪姿态 + 张开夹爪")
    data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
    move_to(model, data, READY_JOINTS, renderer, cam_name,
            stage="[1/5] Ready", steps=300)

    # ── 阶段 2: 预抓取悬停 ──
    pre_grasp = target_pos.copy()
    pre_grasp[2] += PRE_GRASP_HEIGHT

    print(f"  [2/5] 预抓取悬停 → ({pre_grasp[0]:.3f}, {pre_grasp[1]:.3f}, {pre_grasp[2]:.3f})")
    angles, ok = solve_ik(model, pre_grasp, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（预抓取位置不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[2/5] Pre-grasp", steps=500)

    # ── 阶段 3: 下降 ──
    grasp_pos = target_pos.copy()
    grasp_pos[2] += GRASP_HEIGHT_OFFSET

    print(f"  [3/5] 下降 → ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
    angles, ok = solve_ik(model, grasp_pos, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（抓取位置不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[3/5] Descend", steps=400)

    # ── 阶段 4: 闭合夹爪 ──
    print("  [4/5] 闭合夹爪")
    data.ctrl[GRIPPER_IDX] = GRIPPER_CLOSE
    # 保持臂体不动，等夹爪闭合
    move_to(model, data, data.ctrl[:N_ARM_JOINTS].copy(), renderer, cam_name,
            stage="[4/5] Grasping", steps=300)

    # ── 阶段 5: 抬起 ──
    lift_pos = target_pos.copy()
    lift_pos[2] += LIFT_HEIGHT

    print(f"  [5/5] 抬起 → ({lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f})")
    angles, ok = solve_ik(model, lift_pos, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（抬起位置不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[5/5] Lifting", steps=500)

    # 最终展示
    render_frame(model, data, renderer, cam_name, stage="Done!")
    return True


# ═══════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MuJoCo 抓取控制 Pipeline")
    parser.add_argument("--model", type=str, default=None,
                        help="YOLO 权重路径")
    parser.add_argument("--target", type=str, default=None,
                        help="指定目标 (如 blue_cube)，不指定则抓置信度最高的")
    parser.add_argument("--camera", type=str, default="camera_front",
                        help="渲染相机 (camera_front / camera_top / camera_side)")
    args = parser.parse_args()

    from ultralytics import YOLO

    # ── 查找 YOLO 模型 ──
    model_path = args.model
    if model_path is None:
        for p in [
            os.path.join(_SCRIPT_DIR, "runs", "detect", "train", "weights", "best.pt"),
            os.path.join(_SCRIPT_DIR, "best.pt"),
        ]:
            if os.path.exists(p):
                model_path = p
                break
    if model_path is None:
        print("错误: 未找到 YOLO 模型，请指定 --model")
        return

    print(f"YOLO 模型: {model_path}")
    yolo = YOLO(model_path)

    # ── 加载 MuJoCo 场景 ──
    mj_model = mujoco.MjModel.from_xml_path(SCENE_XML)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    # 初始化：夹爪张开
    mj_data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN

    # 物理稳定（让物体落到桌面）
    print("场景初始化...")
    for _ in range(1000):
        mujoco.mj_step(mj_model, mj_data)

    # ── YOLO 检测 ──
    print("\nYOLO 检测中...")
    detections, det_img = detect_objects(yolo, mj_model, mj_data, renderer)

    if not detections:
        print("未检测到任何物体!")
        renderer.close()
        return

    print(f"\n检测到 {len(detections)} 个物体:")
    for name, conf, body, pos in detections:
        print(f"  {name:20s}  conf={conf:.2f}  "
              f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")

    # 显示检测结果
    cv2.imshow(WINDOW, det_img)
    print("\n按任意键开始抓取...")
    cv2.waitKey(0)

    # ── 选择目标 ──
    if args.target:
        target = None
        for d in detections:
            if d[0] == args.target or d[2] == args.target:
                target = d
                break
        if target is None:
            print(f"未检测到指定目标: {args.target}")
            renderer.close()
            return
    else:
        # 选置信度最高的
        target = max(detections, key=lambda x: x[1])

    name, conf, body, pos = target
    print(f"\n{'='*50}")
    print(f"  目标: {name}")
    print(f"  置信度: {conf:.2f}")
    print(f"  位置: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
    print(f"{'='*50}")

    # ── 执行抓取 ──
    print("\n开始抓取序列...\n")
    success = execute_grasp(mj_model, mj_data, pos, renderer, args.camera)

    if success:
        print("\n✓ 抓取完成!")
    else:
        print("\n✗ 抓取失败")

    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    renderer.close()


if __name__ == "__main__":
    main()

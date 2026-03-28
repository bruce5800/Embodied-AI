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
LIFT_HEIGHT = 0.15                   # 抬起高度
MAX_GRASP_ATTEMPTS = 3               # 最大重试次数
GROUND_Z_THRESHOLD = 0.06            # 低于此高度视为仍在地面
# TCP 偏移：从物体中心到实际 IK 目标点的修正 [dx, dy, dz]
# 如果夹爪落点偏了，调这里：
#   dx > 0 → 末端向 +X 移（远离臂基座方向）
#   dy > 0 → 末端向 +Y 移（左右方向）
#   dz > 0 → 末端抬高
GRASP_OFFSET = np.array([0.02, -0.02, 0.02])
READY_JOINTS = [0.0, -0.8, 0.5, 0.3, 0.0]  # 就绪姿态


# ═══════════════════════════════════════════════════════════════
#  IK 求解器 — 基于 MuJoCo Jacobian + 阻尼最小二乘
# ═══════════════════════════════════════════════════════════════

def _ik_once(model, target, site_id, initial_qpos, max_iter=500, tol=1e-3):
    """
    单次 IK 求解（内部函数）。
    从 initial_qpos 出发，用 Jacobian + 阻尼最小二乘迭代。
    """
    data_ik = mujoco.MjData(model)
    data_ik.qpos[:] = initial_qpos[:]

    for _ in range(max_iter):
        mujoco.mj_forward(model, data_ik)

        err = target - data_ik.site_xpos[site_id]
        dist = np.linalg.norm(err)

        if dist < tol:
            return data_ik.qpos[:N_ARM_JOINTS].copy(), dist

        # Jacobian: 3×nv 矩阵
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data_ik, jacp, None, site_id)
        J = jacp[:, :N_ARM_JOINTS]

        # 自适应阻尼：远离目标时大步快走，靠近时小步精调
        lam = max(0.01, min(0.1, dist * 0.5))
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err)

        # 限制单步幅度
        max_step = 0.2
        scale = np.max(np.abs(dq))
        if scale > max_step:
            dq *= max_step / scale

        data_ik.qpos[:N_ARM_JOINTS] += dq

        # 关节限幅
        for j in range(N_ARM_JOINTS):
            lo, hi = model.jnt_range[j]
            data_ik.qpos[j] = np.clip(data_ik.qpos[j], lo, hi)

    mujoco.mj_forward(model, data_ik)
    final_err = np.linalg.norm(target - data_ik.site_xpos[site_id])
    return data_ik.qpos[:N_ARM_JOINTS].copy(), final_err


def solve_ik(model, target_pos, current_qpos=None, max_iter=500, tol=1e-3,
             num_attempts=8):
    """
    使用 MuJoCo 内置 mj_jacSite 求解逆运动学（多次随机重启）。
    仅控制 joint1-5（臂体），不动 joint6（夹爪）。

    原理:
      1. 计算末端当前位置与目标的误差 e
      2. 通过 mj_jacSite 计算末端对关节的 Jacobian J
      3. 阻尼最小二乘: Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
      4. 更新关节角，限幅，重复直到收敛
      5. 多次随机初始猜测，取最优解（避免局部最小值）

    Returns:
        (joint_angles, success)
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    target = np.array(target_pos, dtype=np.float64)

    best_angles = None
    best_err = float('inf')

    # 构建初始猜测列表
    base_qpos = np.zeros(model.nq)
    if current_qpos is not None:
        base_qpos[:] = current_qpos[:]

    for attempt in range(num_attempts):
        init_qpos = base_qpos.copy()

        if attempt == 0:
            # 第 1 次：从当前姿态出发
            pass
        elif attempt == 1:
            # 第 2 次：从就绪姿态出发
            init_qpos[:N_ARM_JOINTS] = READY_JOINTS
        else:
            # 其余：随机采样关节角
            for j in range(N_ARM_JOINTS):
                lo, hi = model.jnt_range[j]
                init_qpos[j] = np.random.uniform(lo, hi)

        angles, err = _ik_once(model, target, site_id, init_qpos,
                               max_iter=max_iter, tol=tol)

        if err < best_err:
            best_err = err
            best_angles = angles

        # 足够精确就提前退出
        if best_err < tol:
            break

    success = best_err < tol * 5
    return best_angles, success


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
    grasp_pos = target_pos + GRASP_OFFSET          # 加 TCP 偏移修正
    pre_grasp = grasp_pos.copy()
    pre_grasp[2] += PRE_GRASP_HEIGHT               # 再抬高悬停

    print(f"  [2/5] 预抓取悬停 → ({pre_grasp[0]:.3f}, {pre_grasp[1]:.3f}, {pre_grasp[2]:.3f})")
    angles, ok = solve_ik(model, pre_grasp, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（预抓取位置不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[2/5] Pre-grasp", steps=500)

    # ── 阶段 3: 下降 ──
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
    lift_pos = grasp_pos.copy()
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


def check_grasp_success(model, data, body_name, renderer, cam_name):
    """
    验证抓取是否成功：检查目标物体是否已离开地面。
    如果物体 z 坐标高于阈值，说明被抬起来了。
    """
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    obj_z = data.xpos[body_id][2]
    lifted = obj_z > GROUND_Z_THRESHOLD

    status = f"物体高度: {obj_z:.3f}m — {'已抬起' if lifted else '仍在地面'}"
    print(f"  验证: {status}")
    render_frame(model, data, renderer, cam_name,
                 stage="OK" if lifted else "MISS", extra_text=status)
    return lifted


def return_to_ready(model, data, renderer, cam_name):
    """回到就绪位置并张开夹爪，准备重试。"""
    data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
    move_to(model, data, READY_JOINTS, renderer, cam_name,
            stage="Retry - Ready", steps=400)
    # 等物体落稳
    for _ in range(500):
        mujoco.mj_step(model, data)


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

    # ── 首次 YOLO 检测 ──
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

    cv2.imshow(WINDOW, det_img)
    print("\n按任意键开始抓取...")
    cv2.waitKey(0)

    # ── 选择目标 ──
    target_name = args.target
    if target_name:
        target = None
        for d in detections:
            if d[0] == target_name or d[2] == target_name:
                target = d
                break
        if target is None:
            print(f"未检测到指定目标: {target_name}")
            renderer.close()
            return
    else:
        target = max(detections, key=lambda x: x[1])
        target_name = target[2]  # body_name

    # ── 抓取 + 验证 + 重试循环 ──
    for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
        name, conf, body, pos = target

        print(f"\n{'='*50}")
        print(f"  第 {attempt}/{MAX_GRASP_ATTEMPTS} 次尝试")
        print(f"  目标: {name}  置信度: {conf:.2f}")
        print(f"  位置: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
        print(f"{'='*50}\n")

        # 执行抓取
        grasp_ok = execute_grasp(mj_model, mj_data, pos, renderer, args.camera)

        if not grasp_ok:
            print(f"\n  第 {attempt} 次: 抓取动作失败（IK 不可达）")
        else:
            # 验证：物体是否被抬起
            if check_grasp_success(mj_model, mj_data, body,
                                   renderer, args.camera):
                print(f"\n✓ 第 {attempt} 次尝试成功!")
                break

            print(f"\n  第 {attempt} 次: 物体未抓住")

        # 还有重试机会 → 回到就绪位，重新检测
        if attempt < MAX_GRASP_ATTEMPTS:
            print("\n  回到就绪位，重新检测...")
            return_to_ready(mj_model, mj_data, renderer, args.camera)

            # 重新检测，获取最新位置（物体可能被碰移位了）
            detections, _ = detect_objects(yolo, mj_model, mj_data, renderer)
            target = None
            for d in detections:
                if d[2] == target_name:
                    target = d
                    break
            if target is None:
                print(f"  重新检测未找到 {target_name}，终止")
                break
        else:
            print(f"\n✗ {MAX_GRASP_ATTEMPTS} 次尝试均失败")

    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    renderer.close()


if __name__ == "__main__":
    main()

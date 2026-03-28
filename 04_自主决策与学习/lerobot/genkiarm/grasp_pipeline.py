"""
MuJoCo 智能分拣 Pipeline
========================
YOLO 检测 → 目标定位 → IK 求解 → 抓取 → 放置到对应区域

流程：
  1. 渲染场景 → YOLO 检测所有物体
  2. 获取目标 3D 位置（仿真中直接读取 body position）
  3. IK 求解：Jacobian 迭代法（使用 MuJoCo 内置 mj_jacSite）
  4. 抓取：就绪 → 预抓取悬停 → 下降 → 闭合夹爪 → 抬起
  5. 放置：移动到对应分拣区上方 → 下降 → 松开夹爪 → 抬起
  6. 循环处理所有检测到的物体

用法：
  python grasp_pipeline.py                              # 分拣所有物体
  python grasp_pipeline.py --target blue_cube           # 只分拣指定物体
  python grasp_pipeline.py --model best.pt --no-place   # 只抓不放
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
PLACE_HEIGHT = 0.08                  # 放置时松手的高度
PLACE_HOVER_HEIGHT = 0.15            # 放置区上方悬停高度
READY_JOINTS = [0.0, -0.8, 0.5, 0.3, 0.0]  # 就绪姿态

# ─── 已知放置区（场景中预定义的区域）───
ZONES = {
    "zone_red":    np.array([-0.15, -0.35, 0.0]),
    "zone_blue":   np.array([-0.15,  0.35, 0.0]),
    "zone_green":  np.array([-0.50,  0.00, 0.0]),
    "zone_yellow": np.array([-0.15,  0.00, 0.0]),
}

# ─── 默认分拣规则（物体 → 放置区名称）───
# 当没有外部指令时使用此映射
DEFAULT_SORT_RULES = {
    "red_cylinder":    "zone_red",
    "blue_cube":       "zone_blue",
    "green_sphere":    "zone_green",
    "yellow_cylinder": "zone_yellow",
}


# ═══════════════════════════════════════════════════════════════
#  放置指令系统 — 为 LLM 接入设计
# ═══════════════════════════════════════════════════════════════
#
#  LLM 只需输出一个 JSON 指令列表，每条指令是一个 dict:
#
#  支持的 mode:
#    "zone"     → 放到预定义区域       {"object": "red_cylinder", "mode": "zone", "zone": "zone_blue"}
#    "absolute" → 放到绝对坐标         {"object": "red_cylinder", "mode": "absolute", "position": [-0.3, 0.1, 0.0]}
#    "relative" → 放到另一物体的相对位置 {"object": "red_cylinder", "mode": "relative",
#                                        "ref_object": "blue_cube", "relation": "above", "offset": 0.08}
#    "line"     → 多个物体排成一排      {"objects": ["red_cylinder", "blue_cube", "green_sphere"],
#                                        "mode": "line", "start": [-0.4, -0.2, 0.0],
#                                        "end": [-0.4, 0.2, 0.0]}
#
#  示例 — LLM 输出:
#    user: "把红色圆柱放到蓝色方块上面"
#    LLM → [{"object": "red_cylinder", "mode": "relative",
#            "ref_object": "blue_cube", "relation": "above", "offset": 0.08}]
#
#    user: "把所有物体排成一排"
#    LLM → [{"objects": ["red_cylinder","blue_cube","green_sphere","yellow_cylinder"],
#            "mode": "line", "start": [-0.35, -0.25, 0.0], "end": [-0.35, 0.25, 0.0]}]

def resolve_place_target(instruction, mj_model, mj_data):
    """
    将一条放置指令解析为具体的 (object_name, [x, y, z]) 列表。

    Args:
        instruction: dict, 放置指令
        mj_model: MuJoCo model (用于查询物体当前位置)
        mj_data:  MuJoCo data

    Returns:
        [(object_name, target_xyz), ...]  一条指令可能产生多个放置任务（如 line 模式）
    """
    mujoco.mj_forward(mj_model, mj_data)
    mode = instruction.get("mode", "zone")

    if mode == "zone":
        # 放到预定义区域
        obj = instruction["object"]
        zone_name = instruction.get("zone", DEFAULT_SORT_RULES.get(obj))
        if zone_name is None or zone_name not in ZONES:
            print(f"  ✗ 未知放置区: {zone_name}")
            return []
        return [(obj, ZONES[zone_name].copy())]

    elif mode == "absolute":
        # 放到绝对坐标
        obj = instruction["object"]
        pos = np.array(instruction["position"], dtype=np.float64)
        return [(obj, pos)]

    elif mode == "relative":
        # 放到另一个物体的相对位置
        obj = instruction["object"]
        ref = instruction["ref_object"]
        relation = instruction.get("relation", "above")
        offset_val = instruction.get("offset", 0.08)

        # 查询参考物体的当前位置
        ref_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, ref)
        if ref_body_id < 0:
            print(f"  ✗ 参考物体 {ref} 不存在")
            return []
        ref_pos = mj_data.xpos[ref_body_id].copy()

        # 根据 relation 计算偏移
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
        # 多个物体排成一排
        objects = instruction["objects"]
        start = np.array(instruction["start"], dtype=np.float64)
        end = np.array(instruction["end"], dtype=np.float64)
        n = len(objects)
        tasks = []
        for i, obj in enumerate(objects):
            t = i / max(n - 1, 1)   # 0 ~ 1 之间均匀分布
            pos = start + t * (end - start)
            tasks.append((obj, pos))
        return tasks

    else:
        print(f"  ✗ 未知放置模式: {mode}")
        return []


def build_default_instructions(detections):
    """
    没有外部 LLM 指令时，使用默认分拣规则生成指令列表。
    """
    instructions = []
    for _name, _conf, body_name, _pos in detections:
        if body_name in DEFAULT_SORT_RULES:
            instructions.append({
                "object": body_name,
                "mode": "zone",
                "zone": DEFAULT_SORT_RULES[body_name],
            })
    return instructions


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


def execute_place(model, data, place_pos, renderer, cam_name="camera_front"):
    """
    将已抓取的物体放到指定坐标。
    假设调用时物体已在夹爪中（刚执行完 execute_grasp 且验证成功）。

    Args:
        place_pos: np.array [x, y, z] — 放置目标坐标（由 resolve_place_target 计算）

    流程：
      阶段 1 — 移动到目标上方悬停
      阶段 2 — 下降到放置高度
      阶段 3 — 松开夹爪
      阶段 4 — 抬起离开
      阶段 5 — 回到就绪位

    Returns:
        success: bool
    """
    zone_pos = np.array(place_pos, dtype=np.float64)

    print(f"\n  放置目标 → ({zone_pos[0]:.3f}, {zone_pos[1]:.3f}, {zone_pos[2]:.3f})")

    # ── 阶段 1: 放置区上方悬停 ──
    hover_pos = zone_pos.copy()
    hover_pos[2] += PLACE_HOVER_HEIGHT

    print(f"  [放置 1/5] 悬停 → ({hover_pos[0]:.3f}, {hover_pos[1]:.3f}, {hover_pos[2]:.3f})")
    angles, ok = solve_ik(model, hover_pos, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（放置区悬停不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[Place 1/5] Hover", steps=500)

    # ── 阶段 2: 下降到放置高度 ──
    place_pos = zone_pos.copy()
    place_pos[2] += PLACE_HEIGHT

    print(f"  [放置 2/5] 下降 → ({place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f})")
    angles, ok = solve_ik(model, place_pos, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（放置位置不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[Place 2/5] Descend", steps=400)

    # ── 阶段 3: 松开夹爪 ──
    print("  [放置 3/5] 松开夹爪")
    data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
    move_to(model, data, data.ctrl[:N_ARM_JOINTS].copy(), renderer, cam_name,
            stage="[Place 3/5] Release", steps=300)

    # ── 阶段 4: 抬起离开 ──
    retreat_pos = place_pos.copy()
    retreat_pos[2] += PLACE_HOVER_HEIGHT

    print(f"  [放置 4/5] 抬起 → ({retreat_pos[0]:.3f}, {retreat_pos[1]:.3f}, {retreat_pos[2]:.3f})")
    angles, ok = solve_ik(model, retreat_pos, data.qpos.copy())
    if not ok:
        print("  ✗ IK 求解失败（抬起不可达）")
        return False
    move_to(model, data, angles, renderer, cam_name,
            stage="[Place 4/5] Retreat", steps=400)

    # ── 阶段 5: 回到就绪位 ──
    print("  [放置 5/5] 回到就绪位")
    move_to(model, data, READY_JOINTS, renderer, cam_name,
            stage="[Place 5/5] Ready", steps=300)

    render_frame(model, data, renderer, cam_name, stage="Placed!")
    return True


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

def pick_and_place_one(mj_model, mj_data, yolo, target_name, place_pos,
                       renderer, cam_name, do_place=True):
    """
    对单个物体执行完整的 抓取(+重试) → 放置 流程。

    Args:
        target_name: 要抓取的物体 body 名称
        place_pos:   放置目标坐标 np.array [x,y,z]（由 resolve_place_target 计算）

    Returns:
        True 如果成功抓取（并放置）
    """
    for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
        # 每次重试前重新检测，拿到最新位置
        detections, _ = detect_objects(yolo, mj_model, mj_data, renderer)
        target = None
        for d in detections:
            if d[2] == target_name:
                target = d
                break
        if target is None:
            print(f"  检测不到 {target_name}，跳过")
            return False

        name, conf, body, pos = target
        place_info = (f"→ ({place_pos[0]:.2f}, {place_pos[1]:.2f}, {place_pos[2]:.2f})"
                      if do_place else "（仅抓取）")
        print(f"\n{'='*50}")
        print(f"  第 {attempt}/{MAX_GRASP_ATTEMPTS} 次尝试")
        print(f"  目标: {name}  置信度: {conf:.2f}")
        print(f"  当前位置: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
        print(f"  放置目标: {place_info}")
        print(f"{'='*50}\n")

        # 抓取
        grasp_ok = execute_grasp(mj_model, mj_data, pos, renderer, cam_name)

        if grasp_ok and check_grasp_success(mj_model, mj_data, body,
                                            renderer, cam_name):
            print(f"\n✓ 抓取成功!")

            # 放置
            if do_place:
                place_ok = execute_place(mj_model, mj_data, place_pos,
                                         renderer, cam_name)
                if place_ok:
                    print(f"✓ {name} 已放置")
                else:
                    print(f"✗ 放置失败，松手释放")
                    mj_data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
                    move_to(mj_model, mj_data, READY_JOINTS, renderer,
                            cam_name, stage="Release", steps=300)
            return True

        # 抓取失败，回到就绪位重试
        print(f"\n  第 {attempt} 次失败")
        if attempt < MAX_GRASP_ATTEMPTS:
            print("  回到就绪位，重新检测...")
            return_to_ready(mj_model, mj_data, renderer, cam_name)

    print(f"\n✗ {target_name}: {MAX_GRASP_ATTEMPTS} 次均失败")
    return False


def execute_task_list(tasks, mj_model, mj_data, yolo, renderer, cam_name,
                      do_place=True):
    """执行一组 (object_name, place_pos) 任务并返回结果。"""
    print(f"\n任务计划 ({len(tasks)} 个):")
    for obj, pos in tasks:
        print(f"  {obj:20s} → ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")

    results = {}
    for i, (obj_name, place_pos) in enumerate(tasks):
        print(f"\n{'#'*50}")
        print(f"  任务 [{i+1}/{len(tasks)}]: {obj_name}")
        print(f"{'#'*50}")

        ok = pick_and_place_one(mj_model, mj_data, yolo, obj_name, place_pos,
                                renderer, cam_name, do_place=do_place)
        results[obj_name] = ok

    # 汇总
    print(f"\n{'='*50}")
    print("  执行结果")
    print(f"{'='*50}")
    for obj_name, ok in results.items():
        status = "✓ 成功" if ok else "✗ 失败"
        print(f"  {obj_name:20s}  {status}")
    success_count = sum(results.values())
    print(f"\n  成功: {success_count}/{len(results)}")
    return results


def main():
    import json

    parser = argparse.ArgumentParser(description="MuJoCo 智能分拣 Pipeline")
    parser.add_argument("--model", type=str, default=None,
                        help="YOLO 权重路径")
    parser.add_argument("--target", type=str, default=None,
                        help="指定目标 (如 blue_cube)，不指定则分拣所有物体")
    parser.add_argument("--camera", type=str, default="camera_front",
                        help="渲染相机 (camera_front / camera_top / camera_side)")
    parser.add_argument("--no-place", action="store_true",
                        help="只抓取不放置（调试用）")
    parser.add_argument("--instructions", type=str, default=None,
                        help="放置指令 JSON 文件路径")
    parser.add_argument("--llm", action="store_true",
                        help="启用 LLM 交互模式（用自然语言下达指令）")
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

    # 初始化
    mj_data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
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

    cv2.imshow(WINDOW, det_img)

    do_place = not args.no_place

    # ═══════════════════════════════════════════════════
    #  LLM 交互模式
    # ═══════════════════════════════════════════════════
    if args.llm:
        from llm_planner import LLMPlanner

        print("\n" + "="*50)
        print("  LLM 交互模式（DeepSeek）")
        print("  输入自然语言指令，如:")
        print('    "把红色圆柱放到蓝色方块上面"')
        print('    "把所有物体排成一排"')
        print('    "把绿色球放到红色区域"')
        print("  输入 quit 退出")
        print("="*50)

        planner = LLMPlanner()

        while True:
            print()
            user_input = input("你想怎么摆？>>> ").strip()
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break

            # 重新检测，获取最新物体位置
            detections, det_img = detect_objects(yolo, mj_model, mj_data, renderer)
            cv2.imshow(WINDOW, det_img)
            cv2.waitKey(1)

            if not detections:
                print("  场景中没有检测到物体")
                continue

            # LLM 解析
            print(f"\n  DeepSeek 解析中...")
            instructions = planner.parse(user_input, scene_objects=detections)

            if not instructions:
                print("  解析失败，请换个说法试试")
                continue

            print(f"\n  LLM 生成 {len(instructions)} 条指令:")
            print(f"  {json.dumps(instructions, indent=2, ensure_ascii=False)}")

            # 解析为具体任务
            tasks = []
            for inst in instructions:
                resolved = resolve_place_target(inst, mj_model, mj_data)
                tasks.extend(resolved)

            if not tasks:
                print("  没有可执行的任务")
                continue

            # 确认执行
            print(f"\n  即将执行 {len(tasks)} 个任务，按任意键开始...")
            cv2.waitKey(0)

            execute_task_list(tasks, mj_model, mj_data, yolo,
                              renderer, args.camera, do_place=do_place)

        print("\nLLM 模式结束")

    # ═══════════════════════════════════════════════════
    #  非 LLM 模式：JSON 文件 / 指定目标 / 默认分拣
    # ═══════════════════════════════════════════════════
    else:
        print("\n按任意键开始分拣...")
        cv2.waitKey(0)

        # 构建放置指令
        if args.instructions:
            with open(args.instructions, "r") as f:
                instructions = json.load(f)
            print(f"\n加载外部指令: {args.instructions} ({len(instructions)} 条)")
        elif args.target:
            instructions = [{"object": args.target, "mode": "zone",
                             "zone": DEFAULT_SORT_RULES.get(args.target)}]
        else:
            instructions = build_default_instructions(detections)

        # 解析 → 执行
        tasks = []
        for inst in instructions:
            resolved = resolve_place_target(inst, mj_model, mj_data)
            tasks.extend(resolved)

        if not tasks:
            print("没有可执行的任务")
        else:
            execute_task_list(tasks, mj_model, mj_data, yolo,
                              renderer, args.camera, do_place=do_place)

    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    renderer.close()


if __name__ == "__main__":
    main()

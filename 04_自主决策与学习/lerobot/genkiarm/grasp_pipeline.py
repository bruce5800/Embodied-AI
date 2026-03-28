"""
MuJoCo 智能分拣 Pipeline — 主入口
==================================
YOLO 检测 → 目标定位 → IK 求解 → 抓取 → 放置

模块结构：
  config.py         — 全局参数（场景、物体、抓取/放置参数）
  ik_solver.py      — 逆运动学求解（MuJoCo Jacobian + DLS）
  motion_control.py — 渲染、运动控制、抓取/放置动作
  task_planner.py   — 放置指令解析（zone/absolute/relative/line）
  llm_planner.py    — DeepSeek API 自然语言 → 指令
  grasp_pipeline.py — 主流程编排（本文件）

用法：
  python grasp_pipeline.py                              # 默认分拣
  python grasp_pipeline.py --target blue_cube           # 指定物体
  python grasp_pipeline.py --llm                        # LLM 交互模式
  python grasp_pipeline.py --instructions tasks.json    # JSON 指令文件
  python grasp_pipeline.py --no-place                   # 只抓不放
"""

import argparse
import json
import os

import cv2
import mujoco
import numpy as np

from config import (
    SCENE_XML, SCRIPT_DIR, GRIPPER_IDX, GRIPPER_OPEN,
    MAX_GRASP_ATTEMPTS, READY_JOINTS, WINDOW, N_ARM_JOINTS,
)
from motion_control import (
    detect_objects, execute_grasp, execute_place,
    check_grasp_success, return_to_ready, move_to, render_frame,
)
from task_planner import resolve_place_target, build_default_instructions


# ═══════════════════════════════════════════════════════════════
#  核心流程
# ═══════════════════════════════════════════════════════════════

def pick_and_place_one(mj_model, mj_data, yolo, target_name, place_pos,
                       renderer, cam_name, do_place=True):
    """
    对单个物体执行 抓取(+重试) → 放置。
    """
    for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
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

        grasp_ok = execute_grasp(mj_model, mj_data, pos, renderer, cam_name)

        if grasp_ok and check_grasp_success(mj_model, mj_data, body,
                                            renderer, cam_name):
            print(f"\n✓ 抓取成功!")
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

        print(f"\n  第 {attempt} 次失败")
        if attempt < MAX_GRASP_ATTEMPTS:
            print("  回到就绪位，重新检测...")
            return_to_ready(mj_model, mj_data, renderer, cam_name)

    print(f"\n✗ {target_name}: {MAX_GRASP_ATTEMPTS} 次均失败")
    return False


def execute_task_list(tasks, mj_model, mj_data, yolo, renderer, cam_name,
                      do_place=True):
    """执行一组任务并汇总结果。"""
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

    print(f"\n{'='*50}")
    print("  执行结果")
    print(f"{'='*50}")
    for obj_name, ok in results.items():
        status = "✓ 成功" if ok else "✗ 失败"
        print(f"  {obj_name:20s}  {status}")
    print(f"\n  成功: {sum(results.values())}/{len(results)}")
    return results


# ═══════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MuJoCo 智能分拣 Pipeline")
    parser.add_argument("--model", type=str, default=None,
                        help="YOLO 权重路径")
    parser.add_argument("--target", type=str, default=None,
                        help="指定目标 (如 blue_cube)")
    parser.add_argument("--camera", type=str, default="camera_front",
                        help="渲染相机")
    parser.add_argument("--no-place", action="store_true",
                        help="只抓取不放置")
    parser.add_argument("--instructions", type=str, default=None,
                        help="放置指令 JSON 文件路径")
    parser.add_argument("--llm", action="store_true",
                        help="启用 LLM 交互模式")
    args = parser.parse_args()

    from ultralytics import YOLO

    # ── YOLO 模型 ──
    model_path = args.model
    if model_path is None:
        for p in [
            os.path.join(SCRIPT_DIR, "runs", "detect", "train", "weights", "best.pt"),
            os.path.join(SCRIPT_DIR, "best.pt"),
        ]:
            if os.path.exists(p):
                model_path = p
                break
    if model_path is None:
        print("错误: 未找到 YOLO 模型，请指定 --model")
        return

    print(f"YOLO 模型: {model_path}")
    yolo = YOLO(model_path)

    # ── MuJoCo 场景 ──
    mj_model = mujoco.MjModel.from_xml_path(SCENE_XML)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

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

    # ══════════════════════════════════════════
    #  LLM 交互模式
    # ══════════════════════════════════════════
    if args.llm:
        from llm_planner import LLMPlanner

        print("\n" + "="*50)
        print("  LLM 交互模式（DeepSeek）")
        print("  输入自然语言指令，如:")
        print('    "把红色圆柱放到蓝色方块上面"')
        print('    "把所有物体排成一排"')
        print("  输入 quit 退出")
        print("="*50)

        planner = LLMPlanner()

        while True:
            print()
            user_input = input("你想怎么摆？>>> ").strip()
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break

            detections, det_img = detect_objects(yolo, mj_model, mj_data, renderer)
            cv2.imshow(WINDOW, det_img)
            cv2.waitKey(1)

            if not detections:
                print("  场景中没有检测到物体")
                continue

            print(f"\n  DeepSeek 解析中...")
            instructions = planner.parse(user_input, scene_objects=detections)

            if not instructions:
                print("  解析失败，请换个说法试试")
                continue

            print(f"\n  LLM 生成 {len(instructions)} 条指令:")
            print(f"  {json.dumps(instructions, indent=2, ensure_ascii=False)}")

            tasks = []
            for inst in instructions:
                resolved = resolve_place_target(inst, mj_model, mj_data)
                tasks.extend(resolved)

            if not tasks:
                print("  没有可执行的任务")
                continue

            print(f"\n  即将执行 {len(tasks)} 个任务，按任意键开始...")
            cv2.waitKey(0)

            execute_task_list(tasks, mj_model, mj_data, yolo,
                              renderer, args.camera, do_place=do_place)

        print("\nLLM 模式结束")

    # ══════════════════════════════════════════
    #  非 LLM 模式
    # ══════════════════════════════════════════
    else:
        print("\n按任意键开始分拣...")
        cv2.waitKey(0)

        if args.instructions:
            with open(args.instructions, "r") as f:
                instructions = json.load(f)
            print(f"\n加载外部指令: {args.instructions} ({len(instructions)} 条)")
        elif args.target:
            from config import DEFAULT_SORT_RULES
            instructions = [{"object": args.target, "mode": "zone",
                             "zone": DEFAULT_SORT_RULES.get(args.target)}]
        else:
            instructions = build_default_instructions(detections)

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

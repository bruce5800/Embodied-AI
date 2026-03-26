#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_number_detection.py

整合流程：拍照 -> 选择ROI -> 图像清理 -> 转20x20灰度 -> 模型推理
复用现有脚本，通过子进程顺序执行，保持参数与默认文件名一致。
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    here = Path(__file__).parent
    py = sys.executable

    parser = argparse.ArgumentParser(description="拍照+ROI选择+处理+20x20+推理 一条龙")
    # 路径配置
    parser.add_argument("--photo", default="photo.jpg", help="拍照输出路径")
    parser.add_argument("--roi", default="roi.jpg", help="ROI输出路径")
    parser.add_argument("--clean", default="roi_clean.jpg", help="清理后二值图输出路径")
    parser.add_argument("--gray-img", default="gray_20x20.png", help="20x20灰度图输出路径")
    parser.add_argument("--gray-txt", default="gray_20x20.txt", help="20x20灰度文本输出路径")
    parser.add_argument("--model", default="mnist_20x20_mlp.joblib", help="模型文件路径")

    # 拍照参数（参考 01_capture_photo.py）
    parser.add_argument("--device", type=int, default=0, help="摄像头索引")
    parser.add_argument("--width", type=int, default=None, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=None, help="摄像头高度")
    parser.add_argument("--warmup", type=float, default=0.8, help="预热秒数")
    parser.add_argument("--show-cam", action="store_true", help="拍照时显示预览窗口")

    # ROI参数（参考 02_set_number_roi.py）
    parser.add_argument("--max-width", type=int, default=1280, help="ROI选择窗口最大宽")
    parser.add_argument("--max-height", type=int, default=900, help="ROI选择窗口最大高")

    # 处理参数（参考 03_process_img.py）
    parser.add_argument("--blur", type=int, default=3, help="高斯模糊核大小")
    parser.add_argument("--adaptive", action="store_true", help="使用自适应阈值")
    parser.add_argument("--block-size", type=int, default=15, help="自适应阈值 block_size")
    parser.add_argument("--C", type=int, default=3, help="自适应阈值 C")
    parser.add_argument("--invert", action="store_true", help="若数字为黑色背景为白色时反相")
    parser.add_argument("--k", type=int, default=3, help="形态学核大小")
    parser.add_argument("--open", type=int, default=1, help="开运算迭代")
    parser.add_argument("--close", type=int, default=1, help="闭运算迭代")
    parser.add_argument("--erode", type=int, default=0, help="额外腐蚀迭代")
    parser.add_argument("--dilate", type=int, default=1, help="额外膨胀迭代")
    parser.add_argument("--shape", type=str, default="ellipse", choices=["ellipse", "rect", "cross"], help="核形状")
    parser.add_argument("--show-process", action="store_true", help="并排预览 原图/二值化/清理后")

    # 20x20转换（参考 04_convert_gray_number.py）
    parser.add_argument("--method", type=str, default="area", choices=["nearest", "linear", "cubic", "area", "lanczos"], help="缩放插值方式")
    parser.add_argument("--invert-gray", action="store_true", help="反相灰度：白变黑、黑变白")

    args = parser.parse_args()

    # 1) 拍照
    cap_cmd = [py, str(here / "01_capture_photo.py"), "--device", str(args.device), "--output", str(here / args.photo), "--warmup", str(args.warmup)]
    if args.width is not None:
        cap_cmd += ["--width", str(args.width)]
    if args.height is not None:
        cap_cmd += ["--height", str(args.height)]
    if args.show_cam:
        cap_cmd += ["--show"]
    print("[1/5] 正在拍照…")
    run_cmd(cap_cmd)

    # 2) 选择ROI
    roi_cmd = [py, str(here / "02_set_number_roi.py"), "--input", str(here / args.photo), "--output", str(here / args.roi), "--max-width", str(args.max_width), "--max-height", str(args.max_height)]
    print("[2/5] 打开ROI选择窗口…")
    run_cmd(roi_cmd)

    # 3) 图像清理（二值化+形态学）
    proc_cmd = [py, str(here / "03_process_img.py"), "--input", str(here / args.roi), "--output", str(here / args.clean), "--blur", str(args.blur), "--block-size", str(args.block_size), "--C", str(args.C), "--k", str(args.k), "--open", str(args.open), "--close", str(args.close), "--erode", str(args.erode), "--dilate", str(args.dilate), "--shape", args.shape]
    if args.adaptive:
        proc_cmd += ["--adaptive"]
    if args.invert:
        proc_cmd += ["--invert"]
    if args.show_process:
        proc_cmd += ["--show"]
    print("[3/5] 正在进行二值化与形态学清理…")
    run_cmd(proc_cmd)

    # 4) 转为20x20灰度并导出文本
    conv_cmd = [py, str(here / "04_convert_gray_number.py"), "--input", str(here / args.clean), "--out-image", str(here / args.gray_img), "--out-text", str(here / args.gray_txt), "--method", args.method]
    if args.invert_gray:
        conv_cmd += ["--invert"]
    print("[4/5] 正在生成20x20灰度图与文本…")
    run_cmd(conv_cmd)

    # 5) 模型推理
    pred_cmd = [py, str(here / "prodicet.py"), "--model", str(here / args.model), "--txt", str(here / args.gray_txt)]
    print("[5/5] 正在进行模型推理…")
    run_cmd(pred_cmd)

    print("流程完成！")


if __name__ == "__main__":
    main()
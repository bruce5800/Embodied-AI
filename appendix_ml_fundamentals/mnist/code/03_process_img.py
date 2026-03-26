#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_process_img.py

先二值化提取白色数字，再做形态学腐蚀/膨胀，让黑更黑、白更白，整体更像手写数字。
- 默认读取 `roi.jpg`，输出清理后的二值图到 `roi_clean.jpg`
- 支持 OTSU 或自适应阈值；支持选择核大小和迭代次数
- 可同时保存原始二值图以便对比
"""

import os
import sys
import argparse
import cv2
import numpy as np


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def binarize_extract_white(img, blur: int = 3, adaptive: bool = False,
                           block_size: int = 15, C: int = 3, invert: bool = False):
    """把白色部分提取为255，其它变为0。
    - blur: 高斯模糊核（奇数，>0时启用），可稳定阈值
    - adaptive: True 用自适应阈值；False 用 OTSU
    - block_size/C: 自适应阈值参数
    - invert: 若数字是黑色而背景是白色时设为 True
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if blur and blur >= 3:
        blur = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    if adaptive:
        th_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   th_type, max(3, block_size | 1), C)
    else:
        th_type = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) | cv2.THRESH_OTSU
        _, bw = cv2.threshold(gray, 0, 255, th_type)
    return bw


def morph_cleanup(bw, k: int = 3, open_iter: int = 1, close_iter: int = 1,
                  erode_extra: int = 0, dilate_extra: int = 1, shape: str = 'ellipse'):
    """对二值图做形态学清理：先开运算去噪，再闭运算补笔画。
    - k: 核大小（奇数）
    - open_iter: 开运算迭代（腐蚀后膨胀）去除白色小噪点
    - close_iter: 闭运算迭代（膨胀后腐蚀）填补白色断裂与孔洞
    - erode_extra/dilate_extra: 额外的腐蚀/膨胀调整“手写”粗细
    - shape: 'ellipse' | 'rect' | 'cross'
    """
    k = k if k % 2 == 1 else k + 1
    if shape == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    elif shape == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if open_iter > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if erode_extra > 0:
        bw = cv2.erode(bw, kernel, iterations=erode_extra)
    if dilate_extra > 0:
        bw = cv2.dilate(bw, kernel, iterations=dilate_extra)
    return bw


def process_image(input_path: str, output_path: str, bin_output: str | None,
                  blur: int, adaptive: bool, block_size: int, C: int, invert: bool,
                  k: int, open_iter: int, close_iter: int, erode_extra: int, dilate_extra: int,
                  shape: str, show: bool) -> None:
    img = cv2.imread(input_path)
    if img is None:
        print(f"无法读取图片: {input_path}")
        sys.exit(1)

    bw = binarize_extract_white(img, blur=blur, adaptive=adaptive,
                                block_size=block_size, C=C, invert=invert)

    cleaned = morph_cleanup(bw, k=k, open_iter=open_iter, close_iter=close_iter,
                            erode_extra=erode_extra, dilate_extra=dilate_extra, shape=shape)

    ensure_dir_for_file(output_path)
    ok = cv2.imwrite(output_path, cleaned)
    if not ok:
        print("保存清理后的二值图失败！")
        sys.exit(1)

    if bin_output:
        ensure_dir_for_file(bin_output)
        cv2.imwrite(bin_output, bw)

    print(f"已保存清理结果到: {os.path.abspath(output_path)}")
    if bin_output:
        print(f"已保存原始二值图到: {os.path.abspath(bin_output)}")

    if show:
        # 并排查看：原图、二值化、清理后
        h = max(img.shape[0], cleaned.shape[0])
        def resize_h(x):
            if x.shape[0] == h:
                return x
            scale = h / x.shape[0]
            return cv2.resize(x, (int(x.shape[1] * scale), h))
        bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        cl_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        vis = cv2.hconcat([resize_h(img), resize_h(bw_bgr), resize_h(cl_bgr)])
        cv2.imshow("Original | Binary | Cleaned", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="先二值化后形态学清理，让数字更像手写且对比更强")
    parser.add_argument("--input", "-i", type=str, default="roi.jpg", help="输入图片路径，默认 roi.jpg")
    parser.add_argument("--output", "-o", type=str, default="roi_clean.jpg", help="输出清理后的二值图路径")
    parser.add_argument("--bin-output", type=str, default=None, help="可选：保存原始二值图路径")
    parser.add_argument("--blur", type=int, default=3, help="高斯模糊核大小(奇数)，默认 3")
    parser.add_argument("--adaptive", action="store_true", help="使用自适应阈值(默认 OTSU)")
    parser.add_argument("--block-size", type=int, default=15, help="自适应阈值 block_size，默认 15")
    parser.add_argument("--C", type=int, default=3, help="自适应阈值 C，默认 3")
    parser.add_argument("--invert", action="store_true", help="若数字为黑色而背景为白色则启用")
    parser.add_argument("--k", type=int, default=3, help="形态学核大小(奇数)，默认 3")
    parser.add_argument("--open", type=int, default=1, help="开运算迭代次数，默认 1")
    parser.add_argument("--close", type=int, default=1, help="闭运算迭代次数，默认 1")
    parser.add_argument("--erode", type=int, default=0, help="额外腐蚀迭代，默认 0")
    parser.add_argument("--dilate", type=int, default=1, help="额外膨胀迭代，默认 1（让笔画更像粉笔）")
    parser.add_argument("--shape", type=str, default="ellipse", choices=["ellipse", "rect", "cross"], help="核形状")
    parser.add_argument("--show", action="store_true", help="并排预览 原图/二值化/清理后")
    args = parser.parse_args()

    process_image(input_path=args.input,
                  output_path=args.output,
                  bin_output=args.bin_output,
                  blur=args.blur,
                  adaptive=args.adaptive,
                  block_size=args.block_size,
                  C=args.C,
                  invert=args.invert,
                  k=args.k,
                  open_iter=args.open,
                  close_iter=args.close,
                  erode_extra=args.erode,
                  dilate_extra=args.dilate,
                  shape=args.shape,
                  show=args.show)


if __name__ == "__main__":
    main()
import os
import cv2
import random
import argparse
import numpy as np
from typing import List

TARGET_SIZE = 640


def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            paths.append(p)
    return sorted(paths)


def resize_to_square(img, size=TARGET_SIZE):
    h, w = img.shape[:2]
    inter = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=inter)


def scale_overlay_to_fit(overlay_rgba, bg_w, bg_h):
    oh, ow = overlay_rgba.shape[:2]
    # 如果覆盖物比背景大，则缩小到背景的 90% 尺寸以内
    scale = min(1.0, 0.9 * min(bg_w / max(ow, 1), bg_h / max(oh, 1)))
    if scale != 1.0:
        new_w = max(1, int(ow * scale))
        new_h = max(1, int(oh * scale))
        overlay_rgba = cv2.resize(overlay_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return overlay_rgba


def rotate_rgba_expand(overlay_rgba, angle_deg: float):
    # 以中心旋转并扩展画布，边界填充为全透明
    h, w = overlay_rgba.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(
        overlay_rgba,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),  # 全透明背景
    )
    return rotated


def compute_bbox_from_alpha(overlay_rgba, threshold: int = 10):
    # 从 alpha 通道计算有效像素的外接矩形
    alpha = overlay_rgba[:, :, 3]
    mask = alpha > threshold
    if not np.any(mask):
        return None  # 完全透明
    ys, xs = np.where(mask)
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    w = max(1, max_x - min_x + 1)
    h = max(1, max_y - min_y + 1)
    return (min_x, min_y, w, h)


def overlay_rgba_on_bgr(background_bgr, overlay_rgba, x, y):
    bh, bw = background_bgr.shape[:2]
    oh, ow = overlay_rgba.shape[:2]

    # 裁剪以防越界
    if x >= bw or y >= bh:
        return background_bgr
    x2 = min(x + ow, bw)
    y2 = min(y + oh, bh)
    ow_clip = x2 - x
    oh_clip = y2 - y
    if ow_clip <= 0 or oh_clip <= 0:
        return background_bgr

    overlay_bgr = overlay_rgba[:oh_clip, :ow_clip, :3].astype(float)
    alpha = overlay_rgba[:oh_clip, :ow_clip, 3].astype(float) / 255.0
    alpha = alpha[..., None]  # (h, w, 1)

    roi = background_bgr[y:y2, x:x2].astype(float)
    comp = overlay_bgr * alpha + roi * (1.0 - alpha)
    background_bgr[y:y2, x:x2] = comp.astype('uint8')
    return background_bgr


def process_texture_folder(texture_dir: str, overlay_path: str, outdir: str, label_outdir: str, class_id: int):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(label_outdir, exist_ok=True)

    # 读取透明 PNG（保留 alpha）
    overlay_rgba = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay_rgba is None:
        raise FileNotFoundError(f"无法读取覆盖图: {overlay_path}")
    if overlay_rgba.shape[2] == 3:
        # 没有 alpha 的情况，创建全不透明 alpha
        alpha = 255 * (overlay_rgba[:, :, :1] * 0 + 1)
        overlay_rgba = cv2.merge([overlay_rgba, alpha])

    paths = list_images(texture_dir)
    if not paths:
        print(f"在 {texture_dir} 未找到图片文件")
        return

    for idx, p in enumerate(paths, start=1):
        bg = cv2.imread(p)
        if bg is None:
            print(f"跳过无法读取的文件: {p}")
            continue

        # 调整背景分辨率为 640x640
        bg = resize_to_square(bg, TARGET_SIZE)
        bh, bw = bg.shape[:2]

        # 随机旋转覆盖图
        angle = random.uniform(0, 360)
        ov_rot = rotate_rgba_expand(overlay_rgba, angle)

        # 确保覆盖物尺寸适配
        ov = scale_overlay_to_fit(ov_rot, bw, bh)
        oh, ow = ov.shape[:2]

        # 随机位置（确保完整贴合在背景内部）
        max_x = max(0, bw - ow)
        max_y = max(0, bh - oh)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # 为标签计算紧致 bbox（基于 alpha 有效区域）
        bbox_local = compute_bbox_from_alpha(ov, threshold=10)
        label_line = None
        if bbox_local is not None:
            bx, by, bw_box, bh_box = bbox_local
            # 转为背景坐标
            left = x + bx
            top = y + by
            width = bw_box
            height = bh_box
            # 转为 YOLO 归一化 (cx, cy, w, h)
            cx = (left + width / 2.0) / bw
            cy = (top + height / 2.0) / bh
            wn = width / bw
            hn = height / bh
            # 限制到 [0,1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            wn = max(0.0, min(1.0, wn))
            hn = max(0.0, min(1.0, hn))
            label_line = f"{class_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}"

        # 合成（考虑 alpha，不覆盖透明区域）
        comp = overlay_rgba_on_bgr(bg, ov, x, y)

        # 保存到输出目录，文件名保持一致
        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(outdir, f"{base}.jpg")
        cv2.imwrite(out_path, comp)

        # 写 YOLO 标签文件（与图像同名 .txt）
        label_path = os.path.join(label_outdir, f"{base}.txt")
        if label_line is not None:
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(label_line + "\n")
        else:
            # 没有有效像素则写空文件，或跳过
            open(label_path, "w").close()

        print(f"[{idx}/{len(paths)}] 保存: {out_path} | 标签: {label_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="将透明 PNG 随机合成到 640x640 背景上并生成 YOLOv8 标签")
    ap.add_argument("--texture_dir", type=str, default="texture", help="背景素材目录")
    ap.add_argument("--overlay", type=str, default="hxs.png", help="透明 PNG 文件路径")
    ap.add_argument("--outdir", type=str, default="train_imgs", help="输出图像目录")
    ap.add_argument("--label_outdir", type=str, default="train_labels", help="YOLO 标签输出目录")
    ap.add_argument("--class_id", type=int, default=0, help="YOLO 类别 ID，默认为 0")
    ap.add_argument("--seed", type=int, default=None, help="随机种子")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    process_texture_folder(
        args.texture_dir,
        args.overlay,
        args.outdir,
        args.label_outdir,
        args.class_id,
    )


if __name__ == "__main__":
    main()
import os
import argparse
import time
import random
from typing import Tuple, List

import numpy as np
import cv2


SIZE = 640


def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)


def rand_color(brightness: Tuple[int, int] = (40, 220)) -> Tuple[int, int, int]:
    low, high = brightness
    return (
        random.randint(low, high),
        random.randint(low, high),
        random.randint(low, high),
    )


def palette_random(n: int = 6) -> List[Tuple[int, int, int]]:
    base = np.array(rand_color((60, 200)))
    pal = []
    for i in range(n):
        jitter = np.random.randint(-40, 40, size=3)
        c = np.clip(base + jitter, 0, 255).astype(np.uint8)
        pal.append((int(c[0]), int(c[1]), int(c[2])))
    random.shuffle(pal)
    return pal


def add_subtle_noise(img: np.ndarray, amount: float = 0.03) -> np.ndarray:
    h, w = img.shape[:2]
    noise = np.random.randn(h, w, 3) * 255 * amount
    out = img.astype(np.float32) + noise.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def add_vignette(img: np.ndarray, strength: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dx = (x - cx) / (w / 2.0)
    dy = (y - cy) / (h / 2.0)
    dist = np.sqrt(dx * dx + dy * dy)
    mask = 1 - np.clip(dist, 0, 1) * strength
    mask = mask.astype(np.float32)
    out = img.astype(np.float32) * mask[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def generate_stripes(size: int = SIZE, orientation: str = None) -> np.ndarray:
    h = w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    palette = palette_random(6)

    if orientation is None:
        orientation = random.choice(["horizontal", "vertical", "diagonal"])

    stripe_width = random.randint(8, 32)

    if orientation == "horizontal":
        y = 0
        color_idx = 0
        while y < h:
            width = stripe_width + random.randint(-4, 8)
            color = palette[color_idx % len(palette)]
            img[y : min(y + width, h), :, :] = color
            y += width
            color_idx += 1
    elif orientation == "vertical":
        x = 0
        color_idx = 0
        while x < w:
            width = stripe_width + random.randint(-4, 8)
            color = palette[color_idx % len(palette)]
            img[:, x : min(x + width, w), :] = color
            x += width
            color_idx += 1
    else:  # diagonal
        # color index decided by (x + y) // stripe_width
        for y in range(h):
            for x in range(w):
                idx = ((x + y) // stripe_width) % len(palette)
                img[y, x] = palette[idx]

    img = add_subtle_noise(img, 0.02)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)
    img = add_vignette(img, strength=0.18)
    return img


def generate_noise_points(size: int = SIZE, density: float = None) -> np.ndarray:
    h = w = size
    bg = np.full((h, w, 3), rand_color((180, 230)), dtype=np.uint8)

    if density is None:
        density = random.uniform(0.01, 0.05)  # fraction of pixels affected

    num_points = int(h * w * density)
    xs = np.random.randint(0, w, size=num_points)
    ys = np.random.randint(0, h, size=num_points)

    for i in range(num_points):
        r = random.randint(1, 3)
        color = rand_color((0, 255))
        cv2.circle(bg, (int(xs[i]), int(ys[i])), r, color, thickness=-1, lineType=cv2.LINE_AA)

    bg = cv2.GaussianBlur(bg, (3, 3), sigmaX=0.6)
    bg = add_vignette(bg, strength=0.15)
    return bg


def generate_multi_octave_texture(size: int = SIZE, octaves: int = 4) -> np.ndarray:
    h = w = size
    acc = np.zeros((h, w), dtype=np.float32)
    total_weight = 0.0
    for i in range(octaves):
        noise = np.random.rand(h, w).astype(np.float32)
        k = 2 * i + 3  # kernel size
        k = k if k % 2 == 1 else k + 1
        blur = cv2.GaussianBlur(noise, (k, k), sigmaX=1.2 + i * 0.8)
        weight = 0.6 / (2 ** i)
        acc += blur * weight
        total_weight += weight
    acc /= max(total_weight, 1e-6)
    acc = np.clip(acc, 0, 1)

    gray = (acc * 255).astype(np.uint8)
    colormap = random.choice([
        cv2.COLORMAP_OCEAN,
        cv2.COLORMAP_BONE,
        cv2.COLORMAP_PINK,
        cv2.COLORMAP_HOT,
        cv2.COLORMAP_TWILIGHT,
        cv2.COLORMAP_TURBO,
    ])
    img = cv2.applyColorMap(gray, colormap)

    img = add_subtle_noise(img, 0.03)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.7)
    img = add_vignette(img, strength=0.2)
    return img


def generate_fabric(size: int = SIZE) -> np.ndarray:
    h = w = size
    y = np.linspace(0, 2 * np.pi, h)
    x = np.linspace(0, 2 * np.pi, w)
    xx, yy = np.meshgrid(x, y)

    fx = random.uniform(6, 12)
    fy = random.uniform(6, 12)
    phase_x = random.uniform(0, 2 * np.pi)
    phase_y = random.uniform(0, 2 * np.pi)

    weave = (np.sin(fx * xx + phase_x) * np.sin(fy * yy + phase_y))
    # add low-frequency irregularity
    irregular = cv2.GaussianBlur(np.random.rand(h, w).astype(np.float32), (21, 21), sigmaX=6.0)
    weave = weave * (0.8 + 0.4 * irregular)

    # normalize to 0..255
    weave_norm = (weave - weave.min()) / (weave.max() - weave.min() + 1e-6)
    gray = (weave_norm * 255).astype(np.uint8)

    # colorize with a fabric-like palette bias
    tint = np.array(rand_color((120, 200)), dtype=np.uint8)
    img = cv2.merge([gray, gray, gray])
    img = (img.astype(np.float32) * 0.6 + tint.astype(np.float32) * 0.4).astype(np.uint8)

    # add thread-like sharpness by directional blur then sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    img = cv2.filter2D(img, -1, kernel)

    # subtle noise and vignette
    img = add_subtle_noise(img, 0.02)
    img = add_vignette(img, strength=0.18)
    return img


def save_img(img: np.ndarray, outdir: str, prefix: str):
    ensure_outdir(outdir)
    ts = int(time.time() * 1000)
    fname = f"{prefix}_{ts}.png"
    path = os.path.join(outdir, fname)
    cv2.imwrite(path, img)
    return path


def generate(style: str, size: int, outdir: str, count: int):
    generators = {
        "stripes": lambda: generate_stripes(size=size),
        "noise": lambda: generate_noise_points(size=size),
        "texture": lambda: generate_multi_octave_texture(size=size),
        "fabric": lambda: generate_fabric(size=size),
    }

    paths = []
    if style == "all":
        keys = list(generators.keys())
        for i in range(count):
            key = keys[i % len(keys)]
            img = generators[key]()
            p = save_img(img, outdir, f"bg_{key}")
            paths.append(p)
    else:
        gen = generators.get(style)
        if gen is None:
            raise ValueError(f"未知风格: {style}. 可选值: stripes, noise, texture, fabric, all")
        for i in range(count):
            img = gen()
            p = save_img(img, outdir, f"bg_{style}")
            paths.append(p)
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="背景图片生成器（640x640，支持多种风格）")
    parser.add_argument("--style", type=str, default="all", choices=["stripes", "noise", "texture", "fabric", "all"], help="生成的背景风格")
    parser.add_argument("--size", type=int, default=SIZE, help="生成图片的大小（正方形边长），默认640")
    parser.add_argument("--count", type=int, default=6, help="生成的数量")
    parser.add_argument("--outdir", type=str, default="backgrounds", help="输出目录")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    paths = generate(args.style, args.size, args.outdir, args.count)
    print("生成完成：")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
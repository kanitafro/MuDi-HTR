# scripts/segment_with_craft.py
"""Segment images into text crops using EasyOCR (preferred) with a simple fallback.

Outputs:
 - crops saved to output directory as crop_000.png, crop_001.png, ...
 - JSON file with bounding boxes and confidences: crops_meta.json

Usage:
  python3 scripts/segment_with_craft.py --image data/processed/xpi03m2uw0a21.jpg --out-dir debug_craft_crops

If EasyOCR is not installed, the script prints install instructions and falls back to simple horizontal projection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("debug_craft_crops"))
    p.add_argument("--expand", type=int, default=8, help="pixels to expand each bbox (both sides)")
    p.add_argument("--min-area", type=int, default=100, help="minimum bbox area to keep")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for EasyOCR if available")
    return p.parse_args()


def save_crop_from_bbox(img_np: np.ndarray, bbox: np.ndarray, out_path: Path):
    # bbox: (4,2) in order [[tl],[tr],[br],[bl]] or similar
    # compute width and height of the target rectangle
    (tl, tr, br, bl) = bbox
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    # destination rectangle
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")

    try:
        import cv2
    except Exception:
        # simple axis-aligned crop fallback
        xs = bbox[:, 0]
        ys = bbox[:, 1]
        x0 = int(max(0, np.floor(xs.min())))
        x1 = int(np.ceil(xs.max()))
        y0 = int(max(0, np.floor(ys.min())))
        y1 = int(np.ceil(ys.max()))
        crop = img_np[y0:y1, x0:x1]
        Image.fromarray((crop * 255).astype('uint8')).save(out_path)
        return

    M = cv2.getPerspectiveTransform(bbox.astype("float32"), dst)
    warped = cv2.warpPerspective((img_np * 255).astype('uint8'), M, (maxW, maxH))
    Image.fromarray(warped).save(out_path)


def horizontal_projection_segment(img_np: np.ndarray, out_dir: Path, expand: int = 8, min_area: int = 100):
    # img_np in 0..1
    H, W = img_np.shape[:2]
    row_mean = img_np.mean(axis=1)
    thr = float(np.clip(row_mean.mean() - 0.25 * row_mean.std(), 0.0, 1.0))
    mask = row_mean < thr
    bands = []
    i = 0
    while i < H:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < H and mask[j]:
            j += 1
        a = max(0, i - 2 - expand)
        b_end = min(H, j + 2 + expand)
        if (b_end - a) * W >= min_area:
            bands.append((a, b_end))
        i = j + 1
    if not bands:
        bands = [(0, H)]

    crops_meta = []
    for idx, (a, b) in enumerate(bands):
        crop = img_np[a:b, :]
        out_path = out_dir / f"crop_{idx:03d}.png"
        Image.fromarray((crop * 255).astype('uint8')).save(out_path)
        crops_meta.append({"id": idx, "bbox": [0, a, W, b], "path": str(out_path), "conf": 1.0})
    return crops_meta


def main():
    args = parse_args()
    img_path = args.image
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pil = Image.open(img_path).convert("L")
    img_np = np.array(pil).astype(np.float32) / 255.0

    try:
        import easyocr
    except Exception:
        print("EasyOCR not available. Falling back to horizontal-projection segmentation.")
        meta = horizontal_projection_segment(img_np, out_dir, expand=args.expand, min_area=args.min_area)
        with open(out_dir / "crops_meta.json", "w", encoding="utf-8") as fh:
            json.dump({"method": "horizontal_projection", "meta": meta}, fh, indent=2)
        print(f"Saved {len(meta)} crops to {out_dir}")
        print("To use EasyOCR install: pip install easyocr opencv-python-headless")
        return

    print("Using EasyOCR to detect text boxes (this may take a few seconds)...")
    reader = easyocr.Reader(["en"], gpu=args.gpu)
    results = reader.readtext(str(img_path), detail=1)

    crops_meta = []
    used = 0
    for idx, (bbox, text, conf) in enumerate(results):
        # bbox is list of four points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        poly = np.array(bbox, dtype="float32")
        # expand polygon by simple bounding-box padding
        xs = poly[:, 0]
        ys = poly[:, 1]
        x0 = int(max(0, np.floor(xs.min()) - args.expand))
        x1 = int(min(img_np.shape[1], np.ceil(xs.max()) + args.expand))
        y0 = int(max(0, np.floor(ys.min()) - args.expand))
        y1 = int(min(img_np.shape[0], np.ceil(ys.max()) + args.expand))
        area = (x1 - x0) * (y1 - y0)
        if area < args.min_area:
            continue
        crop_box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float32")
        out_path = out_dir / f"crop_{used:03d}.png"
        save_crop_from_bbox(img_np, crop_box, out_path)
        crops_meta.append({"id": used, "bbox": [int(x0), int(y0), int(x1), int(y1)], "path": str(out_path), "conf": float(conf), "text": text})
        used += 1

    if used == 0:
        print("EasyOCR found no boxes; falling back to horizontal-projection segmentation.")
        meta = horizontal_projection_segment(img_np, out_dir, expand=args.expand, min_area=args.min_area)
        with open(out_dir / "crops_meta.json", "w", encoding="utf-8") as fh:
            json.dump({"method": "horizontal_projection", "meta": meta}, fh, indent=2)
        print(f"Saved {len(meta)} crops to {out_dir}")
        return

    with open(out_dir / "crops_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"method": "easyocr", "meta": crops_meta}, fh, indent=2)

    print(f"Saved {len(crops_meta)} crops to {out_dir}")


if __name__ == '__main__':
    main()

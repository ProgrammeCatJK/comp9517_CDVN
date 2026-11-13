"""Dataset loading and box utilities. Boxes are pixel (x1, y1, x2, y2)."""
from pathlib import Path

import cv2
import numpy as np

from . import config as C


def image_paths(split):
    return sorted((C.DATA / "images" / split).glob("*.jpg"))


def load_image(path):
    return cv2.imread(str(path))


def load_boxes(split, stem, img_w=512, img_h=512):
    """Ground-truth boxes for one image as an (N, 4) array."""
    path = C.DATA / "labels" / split / f"{stem}.txt"
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32)
    _, xc, yc, w, h = np.array(rows, np.float32).T
    xc, w = xc * img_w, w * img_w
    yc, h = yc * img_h, h * img_h
    return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)


def iou_matrix(a, b):
    """Pairwise IoU between (N, 4) and (M, 4) boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def save_predictions(path: Path, boxes, scores):
    """One line per detection: x1 y1 x2 y2 score (pixels)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {s:.4f}" for (x1, y1, x2, y2), s in zip(boxes, scores)]
    path.write_text("\n".join(lines))


def load_predictions(path: Path):
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if not rows:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    arr = np.array(rows, np.float32)
    return arr[:, :4], arr[:, 4]

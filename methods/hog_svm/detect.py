"""Multi-scale sliding-window detection with a linear SVM on HOG features.

Usage (from the repository root):
    python -m methods.hog_svm.detect --split val
"""
import argparse
import json
import time

import numpy as np
from joblib import Parallel, delayed

from . import config as C
from .data import image_paths, iou_matrix, load_boxes, load_image, save_predictions
from .evaluate import average_precision
from .features import BLOCKS_PER_WINDOW, box_for_window, hog_grid, prepare, resize


def load_model(path=C.MODEL_PATH):
    model = np.load(path)
    return model["w"], float(model["b"])


def scales():
    s = C.SCALE_MAX
    while s >= C.SCALE_MIN:
        yield s
        s /= C.SCALE_STEP


def score_map(grid, w, b):
    """SVM score of every window position, computed as a sum of per-block dot products."""
    n = BLOCKS_PER_WINDOW
    out_r, out_c = grid.shape[0] - n + 1, grid.shape[1] - n + 1
    if out_r <= 0 or out_c <= 0:
        return None
    weights = w.reshape(n, n, -1)
    scores = np.full((out_r, out_c), b, np.float32)
    for i in range(n):
        for j in range(n):
            scores += grid[i:i + out_r, j:j + out_c] @ weights[i, j]
    return scores


def nms(boxes, scores, iou_threshold):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(keep) >= C.MAX_DETECTIONS:
            break
        overlap = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][overlap < iou_threshold]
    return np.array(keep, int)


def detect(image_bgr, w, b, threshold=C.SCORE_THRESHOLD, apply_nms=True):
    """Return (boxes (N, 4), scores (N,)) for one image."""
    image = prepare(image_bgr)
    h, wd = image.shape[:2]
    all_boxes, all_scores = [], []
    for s in scales():
        rw, rh = int(round(wd * s)), int(round(h * s))
        if min(rw, rh) < C.WINDOW:
            break
        scores = score_map(hog_grid(resize(image, rw, rh)), w, b)
        if scores is None:
            continue
        rows, cols = np.nonzero(scores > threshold)
        if not len(rows):
            continue
        sx, sy = rw / wd, rh / h
        side = C.WINDOW / np.sqrt(sx * sy)
        all_boxes.append(box_for_window(cols * C.CELL / sx, rows * C.CELL / sy, side))
        all_scores.append(scores[rows, cols])

    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    boxes = np.concatenate(all_boxes).astype(np.float32)
    scores = np.concatenate(all_scores)
    np.clip(boxes, 0, [wd, h, wd, h], out=boxes)

    top = np.argsort(-scores)[:C.PRE_NMS_TOP_K]
    boxes, scores = boxes[top], scores[top]
    if apply_nms:
        keep = nms(boxes, scores, C.NMS_IOU)
        boxes, scores = boxes[keep], scores[keep]
    return boxes, scores


def _detect_file(path, w, b):
    start = time.perf_counter()
    boxes, scores = detect(load_image(path), w, b)
    return path.stem, boxes, scores, time.perf_counter() - start


def run_split(split, w, b, out_dir=None):
    """Detect on every image of a split; optionally write per-image prediction files."""
    paths = image_paths(split)
    results = Parallel(n_jobs=C.N_JOBS)(delayed(_detect_file)(p, w, b) for p in paths)
    predictions = {stem: (boxes, scores) for stem, boxes, scores, _ in results}
    seconds = [t for *_, t in results]
    if out_dir is not None:
        for stem, (boxes, scores) in predictions.items():
            save_predictions(out_dir / f"{stem}.txt", boxes, scores)
    return predictions, seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    w, b = load_model()
    out_dir = C.OUT / "preds" / args.split
    wall = time.perf_counter()
    predictions, seconds = run_split(args.split, w, b, out_dir)
    wall = time.perf_counter() - wall

    ground_truth = {stem: load_boxes(args.split, stem) for stem in predictions}
    ap, *_ = average_precision(predictions, ground_truth, 0.5)
    metrics = {
        "split": args.split,
        "mAP@0.5": round(ap, 4),
        "images": len(seconds),
        "mean_seconds_per_image": round(float(np.mean(seconds)), 3),
        "wall_seconds": round(wall, 1),
    }
    (C.OUT / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("Predictions written to", out_dir)


if __name__ == "__main__":
    main()

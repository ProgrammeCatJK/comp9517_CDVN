"""Save success and failure examples for the report.

Ground truth is drawn in green, detections above OPERATING_THRESHOLD in red.

Usage (from the repository root, after detect.py has written predictions):
    python -m methods.hog_svm.visualise --split val
"""
import argparse
import json

import cv2
import numpy as np

from . import config as C
from .data import image_paths, load_boxes, load_image, load_predictions
from .evaluate import match_counts

GREEN, RED = (0, 200, 0), (0, 0, 255)


def draw(image, gt, boxes, title):
    out = image.copy()
    for x1, y1, x2, y2 in gt.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2)
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), RED, 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def montage(images, columns=4):
    blank = np.zeros_like(images[0])
    images = images + [blank] * (-len(images) % columns)
    rows = [np.hstack(images[i:i + columns]) for i in range(0, len(images), columns)]
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=4, help="examples per group")
    args = parser.parse_args()

    pred_dir = C.OUT / "preds" / args.split
    if not pred_dir.exists():
        raise SystemExit(f"No predictions in {pred_dir}; run detect.py --split {args.split} first.")

    stats = []
    for path in image_paths(args.split):
        boxes, scores = load_predictions(pred_dir / f"{path.stem}.txt")
        keep = scores > C.OPERATING_THRESHOLD
        boxes, scores = boxes[keep], scores[keep]
        gt = load_boxes(args.split, path.stem)
        tp, fp, fn = match_counts(boxes, scores, gt)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
        stats.append({"stem": path.stem, "path": path, "boxes": boxes, "gt": gt,
                      "tp": tp, "fp": fp, "fn": fn, "f1": f1})

    with_birds = [s for s in stats if len(s["gt"]) >= 2]
    groups = {
        "success": sorted(with_birds, key=lambda s: (-s["f1"], -len(s["gt"])))[:args.count],
        "missed_birds": sorted(stats, key=lambda s: -s["fn"])[:args.count],
        "false_positives": sorted(stats, key=lambda s: -s["fp"])[:args.count],
    }

    out_dir = C.OUT / "figures" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, chosen in groups.items():
        panels = []
        for s in chosen:
            title = f"{s['stem']}  TP {s['tp']}  FP {s['fp']}  FN {s['fn']}"
            panel = draw(load_image(s["path"]), s["gt"], s["boxes"], title)
            cv2.imwrite(str(out_dir / f"{name}_{s['stem']}.png"), panel)
            panels.append(panel)
        cv2.imwrite(str(out_dir / f"{name}.png"), montage(panels))
        summary[name] = [{k: s[k] for k in ("stem", "tp", "fp", "fn")} for s in chosen]

    (out_dir / "examples.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Figures written to", out_dir)


if __name__ == "__main__":
    main()

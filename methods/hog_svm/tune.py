"""Sweep NMS_IOU and the score threshold on the validation split.

HOG scoring runs once per image at the lowest threshold; every setting is then
derived by re-running NMS and filtering by score, which gives the same boxes as
running detect.py with that setting.

Usage (from the repository root):
    python -m methods.hog_svm.tune
"""
import json

import numpy as np
from joblib import Parallel, delayed

from . import config as C
from .data import image_paths, load_boxes, load_image
from .detect import detect, load_model, nms
from .evaluate import average_precision

SCORE_THRESHOLDS = [-1.0, -0.5, 0.0, 0.25, 0.35, 0.45, 0.5, 0.6, 0.75, 1.0, 1.25]
NMS_IOUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _candidates(path, w, b):
    boxes, scores = detect(load_image(path), w, b, threshold=min(SCORE_THRESHOLDS), apply_nms=False)
    return path.stem, boxes, scores


def summarise(predictions, ground_truth):
    ap, precision, recall = average_precision(predictions, ground_truth, 0.5)
    p = float(precision[-1]) if len(precision) else 0.0
    r = float(recall[-1]) if len(recall) else 0.0
    count_errors = [abs(len(predictions[s][0]) - len(ground_truth[s])) for s in ground_truth]
    return {
        "mAP@0.5": round(ap, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "F1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "count_MAE": round(float(np.mean(count_errors)), 3),
        "mean_boxes_per_image": round(float(np.mean([len(v[0]) for v in predictions.values()])), 1),
    }


def main():
    w, b = load_model()
    paths = image_paths("val")
    candidates = Parallel(n_jobs=C.N_JOBS)(delayed(_candidates)(p, w, b) for p in paths)
    ground_truth = {stem: load_boxes("val", stem) for stem, *_ in candidates}

    rows = []
    for nms_iou in NMS_IOUS:
        kept = {}
        for stem, boxes, scores in candidates:
            keep = nms(boxes, scores, nms_iou) if len(boxes) else np.zeros(0, int)
            kept[stem] = (boxes[keep], scores[keep])
        for threshold in SCORE_THRESHOLDS:
            predictions = {s: (bx[sc > threshold], sc[sc > threshold]) for s, (bx, sc) in kept.items()}
            rows.append({"NMS_IOU": nms_iou, "score_threshold": threshold, **summarise(predictions, ground_truth)})
            print(json.dumps(rows[-1]))

    best_map = max(rows, key=lambda r: (r["mAP@0.5"], r["score_threshold"]))
    best_f1 = max(
        (r for r in rows if r["NMS_IOU"] == best_map["NMS_IOU"]),
        key=lambda r: (r["F1"], r["score_threshold"]),
    )
    result = {
        "chosen": {
            "NMS_IOU": best_map["NMS_IOU"],
            "SCORE_THRESHOLD": best_map["score_threshold"],
            "OPERATING_THRESHOLD": best_f1["score_threshold"],
        },
        "best_mAP": best_map,
        "best_F1_at_chosen_NMS": best_f1,
        "sweep": rows,
    }
    C.OUT.mkdir(parents=True, exist_ok=True)
    (C.OUT / "tune_val.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["chosen"], indent=2))
    print("Sweep written to", C.OUT / "tune_val.json")


if __name__ == "__main__":
    main()

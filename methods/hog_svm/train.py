"""Train the linear SVM on HOG features with hard negative mining.

Usage (from the repository root):
    python -m methods.hog_svm.train
"""
import json
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC

from . import config as C
from .data import image_paths, iou_matrix, load_boxes, load_image
from .detect import detect
from .features import box_for_window, patch_feature, prepare, window_for_box, window_patch


def _positives(path):
    image = prepare(load_image(path))
    feats = []
    for box in load_boxes("train", path.stem):
        if np.sqrt((box[2] - box[0]) * (box[3] - box[1])) < C.MIN_BOX:
            continue
        patch = window_patch(image, *window_for_box(box))
        feats.append(patch_feature(patch))
        feats.append(patch_feature(patch[:, ::-1]))
    return feats


def _random_negatives(path, seed):
    rng = np.random.default_rng(seed)
    image = prepare(load_image(path))
    h, w = image.shape[:2]
    gt = load_boxes("train", path.stem)
    feats, tries = [], 0
    while len(feats) < C.NEG_PER_IMAGE and tries < C.NEG_PER_IMAGE * 10:
        tries += 1
        side = float(np.exp(rng.uniform(np.log(C.NEG_MIN_SIDE), np.log(min(C.NEG_MAX_SIDE, w, h)))))
        x1, y1 = rng.uniform(0, w - side), rng.uniform(0, h - side)
        box = box_for_window(x1, y1, side)[None]
        if len(gt) and iou_matrix(box, gt).max() >= C.NEG_IOU:
            continue
        feats.append(patch_feature(window_patch(image, x1, y1, side)))
    return feats


def _hard_negatives(path, w, b):
    raw = load_image(path)
    boxes, scores = detect(raw, w, b, threshold=-1.0, apply_nms=False)
    gt = load_boxes("train", path.stem)
    if len(gt):
        background = iou_matrix(boxes, gt).max(axis=1) < C.NEG_IOU
        boxes, scores = boxes[background], scores[background]
    boxes = boxes[np.argsort(-scores)[:C.HARD_NEG_PER_IMAGE]]
    image = prepare(raw)
    feats = []
    for box in boxes:
        feats.append(patch_feature(window_patch(image, *window_for_box(box))))
    return feats


def _collect(jobs):
    chunks = Parallel(n_jobs=C.N_JOBS)(jobs)
    return [f for chunk in chunks for f in chunk]


def fit(X_pos, X_neg):
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    svm = LinearSVC(C=C.SVM_C, class_weight="balanced", max_iter=5000, random_state=C.SEED)
    svm.fit(X, y)
    return svm.coef_.ravel().astype(np.float32), float(svm.intercept_[0])


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    paths = image_paths("train")
    rng = np.random.default_rng(C.SEED)
    start = time.perf_counter()

    X_pos = np.array(_collect(delayed(_positives)(p) for p in paths))
    X_neg = np.array(_collect(delayed(_random_negatives)(p, C.SEED + i) for i, p in enumerate(paths)))
    print(f"positives {len(X_pos)}, random negatives {len(X_neg)}")
    w, b = fit(X_pos, X_neg)

    history = []
    for round_ in range(1, C.HARD_NEG_ROUNDS + 1):
        subset = rng.choice(len(paths), min(C.HARD_NEG_IMAGES, len(paths)), replace=False)
        hard = _collect(delayed(_hard_negatives)(paths[i], w, b) for i in subset)
        if hard:
            X_neg = np.vstack([X_neg, np.array(hard)])
        print(f"round {round_}: +{len(hard)} hard negatives (total {len(X_neg)})")
        w, b = fit(X_pos, X_neg)
        history.append(len(hard))

    seconds = time.perf_counter() - start
    np.savez(C.MODEL_PATH, w=w, b=b)
    info = {
        "train_seconds": round(seconds, 1),
        "positives": len(X_pos),
        "negatives": len(X_neg),
        "hard_negatives_per_round": history,
        "feature_dim": int(w.size),
    }
    (C.OUT / "train_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print("Model saved to", C.MODEL_PATH)


if __name__ == "__main__":
    main()

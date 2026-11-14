"""Single-class average precision (VOC all-point interpolation)."""
import numpy as np

from .data import iou_matrix


def average_precision(predictions, ground_truth, iou_threshold=0.5):
    """predictions: {stem: (boxes, scores)}; ground_truth: {stem: boxes}.

    Returns (ap, precision, recall) where precision/recall trace the PR curve.
    """
    n_gt = sum(len(b) for b in ground_truth.values())
    records = []
    for stem, (boxes, scores) in predictions.items():
        gt = ground_truth.get(stem, np.zeros((0, 4)))
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        ious = iou_matrix(boxes, gt)
        matched = np.zeros(len(gt), bool)
        for k, s in enumerate(scores):
            hit = False
            if len(gt):
                j = int(np.argmax(np.where(matched, -1, ious[k])))
                if not matched[j] and ious[k, j] >= iou_threshold:
                    matched[j] = hit = True
            records.append((s, hit))

    if not records or n_gt == 0:
        return 0.0, np.array([]), np.array([])
    records.sort(key=lambda r: -r[0])
    hits = np.array([h for _, h in records], float)
    tp = np.cumsum(hits)
    fp = np.cumsum(1 - hits)
    recall = tp / n_gt
    precision = tp / (tp + fp)

    envelope = np.concatenate([[0], precision, [0]])
    for i in range(len(envelope) - 2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i + 1])
    r = np.concatenate([[0], recall, [1]])
    steps = np.nonzero(r[1:] != r[:-1])[0]
    ap = float(np.sum((r[steps + 1] - r[steps]) * envelope[steps + 1]))
    return ap, precision, recall


def match_counts(boxes, scores, gt, iou_threshold=0.5):
    """(true positives, false positives, false negatives) for one image."""
    order = np.argsort(-scores)
    ious = iou_matrix(boxes[order], gt)
    matched = np.zeros(len(gt), bool)
    tp = 0
    for k in range(len(order)):
        if not len(gt):
            break
        j = int(np.argmax(np.where(matched, -1, ious[k])))
        if not matched[j] and ious[k, j] >= iou_threshold:
            matched[j] = True
            tp += 1
    return tp, len(boxes) - tp, len(gt) - tp

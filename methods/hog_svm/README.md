# HOG + linear SVM bird detector

Classical method for the COMP9517 group project. Venkat started it; Swapnav finishes tuning and the write-up.

## What it does

1. Describe each 32×32 window with a HOG feature (Week 3 style).
2. Train a linear SVM: bird windows vs background windows, then two rounds of hard-negative mining.
3. At test time, slide that window over a multi-scale image pyramid and keep boxes above a score, then non-maximum suppression.

Training crops match the sliding-window HOG exactly (checked: max difference 0).

## Setup

Put the Beach Bird dataset in `dataset/` at the repository root (it is git-ignored), so that `dataset/images/train` and `dataset/labels/train` exist. On Windows you can link an existing copy instead of moving it:

```powershell
cmd /c mklink /J dataset "..\Beach Bird Detection Dataset v1.0"
```

Create the environment once:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy opencv-python scikit-image scikit-learn joblib
```

## Run it

From the repository root, with the project venv:

```powershell
python -m methods.hog_svm.train                  # ~10 min on 12 CPU cores
python -m methods.hog_svm.tune                   # threshold sweep on val, ~4 min
python -m methods.hog_svm.detect --split val     # ~3 min
python -m methods.hog_svm.visualise --split val  # success / failure figures
python -m methods.hog_svm.detect --split test    # final evaluation only, ~3 min
python -m methods.hog_svm.visualise --split test
```

- Model: `results/hog_svm/model.npz`
- Training log: `results/hog_svm/train_info.json`
- Threshold sweep: `results/hog_svm/tune_val.json`
- Predictions: `results/hog_svm/preds/<split>/<image>.txt` — each line is `x1 y1 x2 y2 score` in pixels
- Metrics: `results/hog_svm/metrics_<split>.json`
- Figures: `results/hog_svm/figures/<split>/` — ground truth green, detections red

The test split has been evaluated once with the frozen settings (see below). Do not tune anything further.

## Validation results

Trained on all 1,369 training images: 17,000 positive windows (bird crops plus mirrored copies) and 43,380 negatives (27,380 random, 16,000 hard). Training took 578 s on 12 CPU cores.

`tune.py` swept `NMS_IOU` (0.2–0.7) and the score threshold (−1.0 to 1.25) on val:

| Setting | mAP@0.5 | Precision | Recall | F1 | Count MAE | Boxes / image |
|---|---:|---:|---:|---:|---:|---:|
| Original (`NMS_IOU` 0.3, score −0.5) | 0.078 | 0.012 | 0.379 | 0.023 | 193.7 | 200.0 |
| Chosen for mAP (`NMS_IOU` 0.5, score 0.25) | **0.082** | 0.016 | 0.479 | 0.030 | 186.8 | 193.1 |
| Operating point (`NMS_IOU` 0.5, score 0.6) | 0.057 | 0.177 | 0.146 | **0.160** | 5.8 | 5.2 |

- mAP@0.5 is reported from the full ranked list (`SCORE_THRESHOLD`). Precision, recall, counting and figures use `OPERATING_THRESHOLD`, the best-F1 score on val.
- Detection takes about 6.7 s per image on one core (166 s wall time for the 291 val images on 12 cores).
- Raising `NMS_IOU` past 0.5 keeps duplicate boxes and lowers mAP; lowering it suppresses neighbouring birds in flocks.
- The SVM gives positive scores to most background windows: below a score of 0.25 nearly every image hits the 200-box cap, and box counts collapse between 0.35 and 0.6.

Failure patterns visible in the figures:

- Dense flocks of small birds are mostly missed (e.g. 97 of 100 birds in `beach_bird_1658`).
- Reflections of dark birds on wet sand or water are detected as birds, often as nested boxes.
- High-contrast birds fire at several scales, so one bird gets a cluster of overlapping boxes that NMS only partly removes.

## Test results

Run once on 15 Nov 2026 with the settings frozen on val (`NMS_IOU` 0.5, `SCORE_THRESHOLD` 0.25, `OPERATING_THRESHOLD` 0.6):

| Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | Count MAE | s / image (1 core) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Val | 0.082 | 0.023 | 0.177 | 0.145 | 0.160 | 5.8 | 6.7 |
| Test | **0.063** | 0.016 | 0.208 | 0.116 | 0.149 | 5.6 | 7.7 |

The 240 test images took 156 s across 12 cores. The validation failure patterns repeat on test. New on test: bare tree branches against the sky (up to 33 false boxes in a bird-free image), seaweed on sand, and overlapping gulls that are not matched at all (`beach_bird_1664`). Test figures are in `results/hog_svm/figures/test/`.

## Knobs (all in `config.py`)

| Setting | Now | Try if… |
|---|---|---|
| `SCORE_THRESHOLD` | 0.25 | too many false boxes → raise it; missed birds → lower it |
| `OPERATING_THRESHOLD` | 0.6 | used for precision/recall, counting and figures |
| `NMS_IOU` | 0.5 | duplicate boxes on one bird → lower it |
| `SCALE_STEP` | 1.2 | missed sizes → 1.1 (slower) |
| `SVM_C` | 0.01 | underfitting → 0.1 |
| `HARD_NEG_ROUNDS` | 2 | still lots of false positives → 3 |
| `COLOUR` | False | want a second variant → True |

Changing `SCORE_THRESHOLD`, `OPERATING_THRESHOLD` or `NMS_IOU` only needs `detect.py` again. Any other setting needs `train.py` and `tune.py` re-run first.

## Possible extensions

Not run, and not needed for the comparison with YOLO:

- Colour HOG, or a different window size.
- A third round of hard-negative mining (`HARD_NEG_ROUNDS = 3`): both rounds so far hit the 20-per-image cap.

Any such variant has to be selected on val. The test numbers above must not drive the choice.

"""All hyperparameters for the HOG + linear SVM bird detector."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"
OUT = ROOT / "results" / "hog_svm"
MODEL_PATH = OUT / "model.npz"

SEED = 0
N_JOBS = -1

# HOG descriptor. WINDOW must be a multiple of CELL.
WINDOW = 32
CELL = 4
BLOCK = 2
ORIENTATIONS = 9
COLOUR = False

# Training samples
CONTEXT = 0.15          # window side = bird side * (1 + CONTEXT)
MIN_BOX = 10            # skip ground-truth birds smaller than this (sqrt of area, px)
NEG_PER_IMAGE = 20
NEG_MIN_SIDE = 12
NEG_MAX_SIDE = 400
NEG_IOU = 0.3           # a window counts as background below this IoU with every bird
SVM_C = 0.01

# Hard negative mining
HARD_NEG_ROUNDS = 2
HARD_NEG_IMAGES = 400
HARD_NEG_PER_IMAGE = 20

# Detection
SCALE_MAX = 2.4         # 2.4x upscale lets a 32px window find ~12px birds
SCALE_MIN = 0.08
SCALE_STEP = 1.2
SCORE_THRESHOLD = -0.5
PRE_NMS_TOP_K = 3000
NMS_IOU = 0.3
MAX_DETECTIONS = 200

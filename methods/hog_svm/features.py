"""HOG features for training patches and for whole images.

Training patches are cropped with a one-cell margin and the outer ring of blocks is
discarded, so patch descriptors match the blocks of a whole-image HOG grid exactly
(skimage zeroes gradients on the image border).
"""
import cv2
import numpy as np
from skimage.feature import hog

from . import config as C

CELLS_PER_WINDOW = C.WINDOW // C.CELL
BLOCKS_PER_WINDOW = CELLS_PER_WINDOW - C.BLOCK + 1


def prepare(image_bgr):
    if C.COLOUR:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255


def resize(image, width, height):
    shrinking = width < image.shape[1]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


def hog_grid(image):
    """HOG blocks of a whole image, shape (block_rows, block_cols, block_dim)."""
    blocks = hog(
        image,
        orientations=C.ORIENTATIONS,
        pixels_per_cell=(C.CELL, C.CELL),
        cells_per_block=(C.BLOCK, C.BLOCK),
        block_norm="L2-Hys",
        feature_vector=False,
        channel_axis=-1 if C.COLOUR else None,
    )
    return blocks.reshape(blocks.shape[0], blocks.shape[1], -1)


def window_patch(image, x1, y1, side):
    """Square region (x1, y1, side) resized to WINDOW px plus a one-cell margin."""
    margin = side * C.CELL / C.WINDOW
    x0, y0 = int(round(x1 - margin)), int(round(y1 - margin))
    size = max(int(round(side + 2 * margin)), 1)
    h, w = image.shape[:2]
    pad = max(0, -x0, -y0, x0 + size - w, y0 + size - h)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        x0, y0 = x0 + pad, y0 + pad
    out = C.WINDOW + 2 * C.CELL
    return resize(image[y0:y0 + size, x0:x0 + size], out, out)


def patch_feature(patch):
    b = BLOCKS_PER_WINDOW
    return hog_grid(patch)[1:1 + b, 1:1 + b].ravel()


def window_for_box(box):
    """Square sampling window (x1, y1, side) around a box, including context."""
    x1, y1, x2, y2 = box
    side = np.sqrt((x2 - x1) * (y2 - y1)) * (1 + C.CONTEXT)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return cx - side / 2, cy - side / 2, side


def box_for_window(x1, y1, side):
    """Inverse of window_for_box: the bird box predicted by a window."""
    inner = side / (1 + C.CONTEXT)
    offset = (side - inner) / 2
    return np.stack([x1 + offset, y1 + offset, x1 + offset + inner, y1 + offset + inner], axis=-1)

"""Click-guided surface outlines from the pinned CPU EdgeSAM models."""
from functools import lru_cache
from hashlib import sha256
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from renderer.floor_detection import outlines_from_mask


_lock = Lock()
_embedding = None


@lru_cache(maxsize=1)
def _sessions():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return tuple(ort.InferenceSession(
        f"/opt/models/edge_sam_3x_{kind}.onnx", sess_options=options,
        providers=["CPUExecutionProvider"])
        for kind in ("encoder", "decoder"))


def select_surface(base_bgr: np.ndarray, points: list[list[float]],
                   labels: list[int]) -> dict:
    """Return a boundary and object holes in original photo coordinates.

    Points are [x, y] pixels; label 1 includes a surface and 0 excludes an
    object. One embedding is cached for repeated clicks on the current photo.
    """
    global _embedding
    if (not isinstance(base_bgr, np.ndarray) or base_bgr.ndim != 3
            or base_bgr.shape[2] != 3 or min(base_bgr.shape[:2]) < 2
            or base_bgr.dtype != np.uint8):
        raise ValueError("The photo must be a nonempty uint8 BGR image.")
    height, width = base_bgr.shape[:2]
    coords = np.asarray(points, dtype=np.float32)
    point_labels = np.asarray(labels, dtype=np.float32)
    if (coords.ndim != 2 or coords.shape[1] != 2 or not len(coords)
            or point_labels.shape != (len(coords),)
            or not np.isfinite(coords).all()
            or not np.isin(point_labels, [0, 1]).all()
            or not (point_labels == 1).any()
            or (coords < 0).any() or (coords[:, 0] >= width).any()
            or (coords[:, 1] >= height).any()):
        raise ValueError("Use photo coordinates and 0/1 labels with at least one include point.")

    scale = 1024 / max(width, height)
    work = (max(2, round(width * scale)), max(2, round(height * scale)))
    resized_coords = coords * np.float32([work[0] / width, work[1] / height])
    key = (base_bgr.shape, sha256(np.ascontiguousarray(base_bgr)).digest())
    with _lock:
        encoder, decoder = _sessions()
        if _embedding is None or _embedding[0] != key:
            rgb = Image.fromarray(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB))
            pixels = np.asarray(rgb.resize(work, Image.Resampling.BILINEAR))
            pixels = pixels.transpose(2, 0, 1)[None].astype(np.float32)
            pixels = ((pixels - np.float32([123.675, 116.28, 103.53])[None, :, None, None])
                      / np.float32([58.395, 57.12, 57.375])[None, :, None, None])
            pixels = np.pad(pixels, ((0, 0), (0, 0),
                                    (0, 1024 - work[1]), (0, 1024 - work[0])))
            features = encoder.run(None, {"image": pixels})[0]
            _embedding = (key, features)
        scores, low_masks = decoder.run(None, {
            "image_embeddings": _embedding[1],
            "point_coords": resized_coords[None],
            "point_labels": point_labels[None],
        })

    # Require an include point before comparing click agreement and stability.
    sample_x = np.clip(np.rint(resized_coords[:, 0]).astype(int), 0, work[0] - 1)
    sample_y = np.clip(np.rint(resized_coords[:, 1]).astype(int), 0, work[1] - 1)
    best_mask, best_rank = None, None
    for score, low_mask in zip(scores[0], low_masks[0]):
        logits = cv2.resize(low_mask, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask = logits[:work[1], :work[0]] > 0
        _, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        included = components[sample_y[point_labels == 1], sample_x[point_labels == 1]]
        roots = set(included) - {0}
        if roots:
            root = max(roots, key=lambda item: (
                np.count_nonzero(included == item), stats[item, cv2.CC_STAT_AREA]))
            mask = components == root
        else:
            mask[:] = False
        matches = int(np.count_nonzero(mask[sample_y, sample_x] == point_labels))
        rank = (bool(roots), matches, float(score))
        if best_rank is None or rank > best_rank:
            best_mask, best_rank = mask, rank
    try:
        return outlines_from_mask(best_mask.astype(np.uint8) * 255, (width, height))
    except ValueError as error:
        raise ValueError("No surface was selected. Add an include point inside the surface.") from error

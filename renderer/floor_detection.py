"""Editable floor outlines from the pinned SegFormer B4 ADE20K model."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


@lru_cache(maxsize=1)
def _session():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(Path("/opt/models/segformer-b4.onnx")), sess_options=options,
        providers=["CPUExecutionProvider"])


def model_input_size(width: int, height: int, long_side: int = 640) -> tuple[int, int]:
    """Photo aspect ratio at the model's scale, in multiples of 32."""
    scale = long_side / max(width, height)
    return (max(32, round(width * scale / 32) * 32),
            max(32, round(height * scale / 32) * 32))


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int,
                  eps: float) -> np.ndarray:
    """He et al. guided filter with a scalar or three-channel colour guide."""
    def box(values):
        return cv2.boxFilter(values, -1, (2 * radius + 1, 2 * radius + 1))

    mean_i, mean_p = box(guide), box(src)
    if guide.ndim == 3:
        covariance = np.stack([
            box(guide[..., i] * guide[..., j]) - mean_i[..., i] * mean_i[..., j]
            for i in range(3) for j in range(3)
        ], axis=-1).reshape(*src.shape, 3, 3)
        cross = box(guide * src[..., None]) - mean_i * mean_p[..., None]
        a = np.linalg.solve(covariance + eps * np.eye(3, dtype=np.float32),
                            cross[..., None])[..., 0]
        b = mean_p - (a * mean_i).sum(axis=-1)
        return (box(a) * guide).sum(axis=-1) + box(b)
    a = (box(guide * src) - mean_i * mean_p) / (box(guide * guide) - mean_i ** 2 + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def floor_mask_from_logits(logits: np.ndarray, size: tuple[int, int],
                           guide_bgr: np.ndarray | None = None) -> np.ndarray:
    """Floor-vs-best-other margin, upsampled to size and snapped to guide edges.

    The model predicts at 1/4 of its input, so its floor edge can sit a whole
    cell (~25 photo px) off the skirting board. The guided filter moves the
    edge onto the strongest nearby edge in the photo.
    """
    others = np.delete(logits, 3, axis=0).max(axis=0)
    margin = cv2.resize(logits[3] - others, size, interpolation=cv2.INTER_LINEAR)
    floor = 1.0 / (1.0 + np.exp(-np.clip(margin, -30, 30)))
    # Refinement must retain regions the model confidently identified as objects.
    possible_floor = floor >= 0.1
    if guide_bgr is not None:
        colour = guide_bgr.astype(np.float32) / 255.0
        cell = max(size[0] / logits.shape[2], size[1] / logits.shape[1])
        floor = guided_filter(colour, floor.astype(np.float32), max(2, round(cell * 3)), 1e-3)
    return ((floor > 0.5) & possible_floor).astype(np.uint8) * 255


def outlines_from_mask(mask: np.ndarray, image_size: tuple[int, int]) -> dict:
    """Trace the dominant floor region and its object holes in photo pixels."""
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    roots = [i for i in range(len(contours)) if hierarchy[0, i, 3] < 0]
    if not roots:
        raise ValueError("No floor was detected. Use Floor outline to select it manually.")
    largest = max(roots, key=lambda i: cv2.contourArea(contours[i]))
    if cv2.contourArea(contours[largest]) < mask.size * 0.005:
        raise ValueError("No floor was detected. Use Floor outline to select it manually.")
    width, height = image_size
    scale = np.array([(width - 1) / (mask.shape[1] - 1),
                      (height - 1) / (mask.shape[0] - 1)])

    def points(contour):
        simple = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
        return np.rint(simple * scale).astype(int).tolist()

    exclusions = []
    child = hierarchy[0, largest, 2]
    while child >= 0:
        # Smaller holes are filter speckle, not objects worth protecting.
        if cv2.contourArea(contours[child]) >= mask.size * 0.0005:
            hole = points(contours[child])
            if len(hole) >= 3:
                exclusions.append(hole)
        child = hierarchy[0, child, 0]
    boundary = points(contours[largest])
    if len(boundary) < 3:
        raise ValueError("No floor was detected. Use Floor outline to select it manually.")
    return {"floor_boundary": boundary, "exclude": exclusions}


def detect_floor(base_bgr: np.ndarray) -> dict:
    height, width = base_bgr.shape[:2]
    rgb = Image.fromarray(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB))
    small = np.asarray(rgb.resize(model_input_size(width, height),
                                  Image.Resampling.BILINEAR))
    pixels = small.astype(np.float32) / 255.0
    pixels = ((pixels - [0.485, 0.456, 0.406]) /
              [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
    logits = _session().run(None, {"pixel_values": pixels})[0][0]
    scale = min(1.0, 1024 / max(width, height))
    work = (round(width * scale), round(height * scale))
    guide = cv2.resize(base_bgr, work, interpolation=cv2.INTER_AREA)
    return outlines_from_mask(floor_mask_from_logits(logits, work, guide),
                              (width, height))

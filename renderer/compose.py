"""Warp the tile sheet into the photo and carry the room's own light onto it."""
import cv2
import numpy as np

from renderer.geometry import Quad, quad_mask_of, sheet_to_photo_homography


def soft_clip(v: np.ndarray, knee: float = 0.80) -> np.ndarray:
    """Roll off highlights. Hard-clipping at 255 flattens the veining away --
    a bright texture times a gain of 1.75 exceeded 255 and erased the pattern.
    """
    x = v / 255.0
    over = x > knee
    x[over] = knee + (1 - knee) * (1 - np.exp(-(x[over] - knee) / (1 - knee)))
    return np.clip(x * 255.0, 0, 255)


def lighting_gain(base_bgr: np.ndarray, mask: np.ndarray, sigma: float = 41.0,
                  lo: float = 0.55, hi: float = 1.35) -> np.ndarray:
    """Floor-only luminance, with restrained detail from real dark shadows.

    Normalised convolution prevents walls and excluded objects from lighting
    the floor. Both scales smooth away grain; the finer one restores only half
    of the dark residual, so contact shadows survive without copying texture.
    """
    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    selected = mask > 127
    if not selected.any():
        return np.ones_like(gray)
    weights = selected.astype(np.float32)

    def smooth(scale: float) -> np.ndarray:
        support = cv2.GaussianBlur(weights, (0, 0), scale)
        luminance = cv2.GaussianBlur(gray * weights, (0, 0), scale)
        return luminance / np.maximum(support, 1e-6)

    low = smooth(sigma)
    detail = smooth(max(2.0, sigma / 8.0))
    shadow = np.minimum(detail / np.maximum(low, 1e-6), 1.0)
    gain = low / max(float(low[selected].mean()), 1e-6)
    return np.clip(gain * (0.5 + 0.5 * shadow), lo, hi)


def render(base_bgr: np.ndarray, sheet: np.ndarray, quad: Quad,
           mask: np.ndarray | None = None, strength: float = 1.0,
           feather: float = 2.0) -> np.ndarray:
    height, width = base_bgr.shape[:2]
    warped = cv2.warpPerspective(
        sheet, sheet_to_photo_homography(sheet.shape, quad), (width, height),
        flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    if mask is None:
        mask = quad_mask_of(base_bgr.shape, quad)

    gain = lighting_gain(base_bgr, mask, sigma=max(9.0, min(height, width) / 18.0))
    lit = soft_clip(warped.astype(np.float32) * (gain[..., None] ** strength))
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather) / 255.0
    alpha[mask == 0] = 0
    alpha = alpha[..., None]
    out = base_bgr.astype(np.float32) * (1 - alpha) + lit * alpha
    return out.astype(np.uint8)

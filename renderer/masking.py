"""Which pixels are floor, and which are things standing on it.

Hand-drawn exclusion is the primary mechanism; the colour model is a
convenience for high-contrast objects only. Spec section 5 measured that no
single colour threshold can keep a near-neutral object AND the floor, which is
why white sanitaryware rooms are routed to pre-built sample scenes instead.
"""
import cv2
import numpy as np

from renderer.geometry import Quad, near_edge_band, quad_mask_of


def drop_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def floor_mask_by_colour_deviation(base_bgr: np.ndarray, quad: Quad,
                                   quad_mask: np.ndarray, threshold: float = 13.0,
                                   lightness_weight: float = 0.15,
                                   min_object_area: int = 400) -> np.ndarray:
    """One-class model: learn floor colour from the near-edge strip, then mark
    deviating pixels as objects.

    Lightness is weighted right down because a cast shadow moves L a long way
    while leaving hue alone. Measured on the test room: lit floor d2=0.3,
    shadowed floor d2=0.5, wooden cabinet d2=19-23.

    Morphology runs on the OBJECTS, never on the floor -- closing the floor
    mask let a few floor-looking pixels inside an object dilate outward and
    swallow the whole object (measured 12% -> 91%).
    """
    lab = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    seed = near_edge_band(quad, quad_mask) > 0
    if seed.sum() < 200:
        return quad_mask
    samples = lab[seed]
    mu = samples.mean(0)
    sd = np.maximum(samples.std(0), 2.0)
    weights = np.array([lightness_weight, 1.0, 1.0], np.float32)
    d2 = (((lab - mu) / sd) ** 2 * weights).sum(-1)

    objects = (((d2 >= threshold) & (quad_mask > 0)) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    objects = cv2.morphologyEx(objects, cv2.MORPH_CLOSE, kernel)
    objects = drop_small_components(objects, min_object_area)
    return cv2.bitwise_and(quad_mask, cv2.bitwise_not(objects))


def build_mask(base_bgr: np.ndarray, quad: Quad,
               exclude: list[list[tuple[float, float]]] | None = None,
               auto: str = "none", threshold: float = 13.0,
               floor_boundary: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Perspective plane clipped to the floor outline, then objects removed."""
    quad_mask = quad_mask_of(base_bgr.shape, quad)
    if floor_boundary is not None:
        boundary_mask = np.zeros(base_bgr.shape[:2], np.uint8)
        cv2.fillPoly(boundary_mask, [np.int32(floor_boundary)], 255)
        quad_mask = cv2.bitwise_and(quad_mask, boundary_mask)
    mask = quad_mask.copy()
    for poly in (exclude or []):
        cv2.fillPoly(mask, [np.int32(poly)], 0)

    if auto == "colour-deviation":
        mask = cv2.bitwise_and(mask, floor_mask_by_colour_deviation(
            base_bgr, quad, quad_mask, threshold=threshold))
    elif auto != "none":
        raise ValueError(f"unknown auto mask mode: {auto!r}")
    return mask

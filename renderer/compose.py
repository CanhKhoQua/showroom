"""Warp the tile sheet into the photo and carry the room's own light onto it."""
import cv2
import numpy as np

from renderer.geometry import Quad, quad_mask_of, sheet_to_photo_homography


_SRGB = np.arange(256, dtype=np.float32) / 255.0
_SRGB_TO_LINEAR = np.where(_SRGB <= 0.04045, _SRGB / 12.92,
                           ((_SRGB + 0.055) / 1.055) ** 2.4)


def soft_clip(v: np.ndarray, knee: float = 0.80) -> np.ndarray:
    """Roll off highlights. Hard-clipping at 255 flattens the veining away --
    a bright texture times a gain of 1.75 exceeded 255 and erased the pattern.
    """
    x = v / 255.0
    over = x > knee
    x[over] = knee + (1 - knee) * (1 - np.exp(-(x[over] - knee) / (1 - knee)))
    return np.clip(x * 255.0, 0, 255)


def lighting_gain(base_bgr: np.ndarray, mask: np.ndarray, sigma: float = 41.0,
                  lo: float = 0.02, hi: float = 1.75) -> np.ndarray:
    """Relative linear irradiance, excluding objects and old floor grain.

    Normalised convolution prevents walls and excluded objects from lighting
    the floor. A fine, still grain-free scale restores dark contact shadows.
    The low floor allows shadows to approach black instead of filling them in.
    """
    gray = (_SRGB_TO_LINEAR[base_bgr[..., 0]] * 0.0722
            + _SRGB_TO_LINEAR[base_bgr[..., 1]] * 0.7152
            + _SRGB_TO_LINEAR[base_bgr[..., 2]] * 0.2126)
    selected = mask > 0
    if not selected.any():
        return np.ones_like(gray)
    weights = mask.astype(np.float32) / 255.0

    def smooth(scale: float) -> np.ndarray:
        support = cv2.GaussianBlur(weights, (0, 0), scale,
                                   borderType=cv2.BORDER_CONSTANT)
        luminance = cv2.GaussianBlur(gray * weights, (0, 0), scale,
                                     borderType=cv2.BORDER_CONSTANT)
        return luminance / np.maximum(support, 1e-6)

    low = smooth(sigma)
    reference = float(low[selected].mean())
    if reference < 1e-6:
        return np.ones_like(gray)
    detail = smooth(max(2.0, min(3.0, sigma / 16.0)))
    shadow = np.minimum(detail / np.maximum(low, 1e-6), 1.0)
    return np.clip(low / reference * shadow, lo, hi)


def _surface_light(base: np.ndarray, mask: np.ndarray, surface_finish: str):
    """Estimate relative light colour and broad highlights, not old albedo.

    Per-channel normalisation removes the old surface's constant colour.
    Masked filtering excludes furniture before downsampling; the small light
    maps limit CPU/memory use. A single photo cannot recover true reflections.
    """
    height, width = mask.shape
    scale = min(1.0, 512 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    weights = mask.astype(np.float32) / 255.0
    support = cv2.resize(weights, size, interpolation=cv2.INTER_AREA)
    sigma = max(3.0, min(size) / 18.0)

    def smooth(values, radius):
        denominator = cv2.GaussianBlur(support, (0, 0), radius,
                                       borderType=cv2.BORDER_CONSTANT)
        numerator = cv2.GaussianBlur(values, (0, 0), radius,
                                     borderType=cv2.BORDER_CONSTANT)
        return numerator / np.maximum(denominator, 1e-6)

    light = np.empty((*support.shape, 3), np.float32)
    luminance = np.zeros_like(support)
    for channel, coefficient in enumerate((0.0722, 0.7152, 0.2126)):
        values = cv2.resize(_SRGB_TO_LINEAR[base[..., channel]] * weights,
                            size, interpolation=cv2.INTER_AREA)
        mean = float(values.sum() / max(float(support.sum()), 1e-6))
        light[..., channel] = smooth(values, sigma) / mean if mean > 1e-6 else 1.0
        luminance += values * coefficient
    neutral = light @ np.array([0.0722, 0.7152, 0.2126], np.float32)
    colour = np.sqrt(np.clip(light / np.maximum(neutral[..., None], 1e-6), 0.65, 1.45))

    highlights = np.zeros_like(support)
    if surface_finish != 'matte':
        radius = sigma if surface_finish == 'satin' else max(1.0, sigma / 3)
        fine = smooth(luminance, radius)
        broad = smooth(luminance, sigma * 3)
        reference = float(luminance.sum() / max(float(support.sum()), 1e-6))
        highlights = np.clip((fine - broad) / max(reference, 1e-6) - 0.08, 0, 1)
        highlights *= 0.06 if surface_finish == 'satin' else 0.18
    return colour, highlights


def render(base_bgr: np.ndarray, sheet: np.ndarray, quad: Quad,
           mask: np.ndarray | None = None, strength: float = 1.0,
           feather: float = 2.0, surface_finish: str = 'matte',
           reflectance: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = quad_mask_of(base_bgr.shape, quad)
    x, y, width, height = cv2.boundingRect(mask)
    out = base_bgr.copy()
    if width == 0 or height == 0:
        return out

    # Keep full-size uint8 output, but warp and allocate float light/color only
    # inside the selected surface. Translating the quad preserves perspective.
    base = base_bgr[y:y + height, x:x + width]
    mask = mask[y:y + height, x:x + width]
    local_quad = [(qx - x, qy - y) for qx, qy in quad]
    homography = sheet_to_photo_homography(sheet.shape, local_quad)
    lit = np.empty((height, width, 3), np.float32)
    # Interpolate light across texture edges without a float RGB sheet copy.
    for channel in range(3):
        lit[..., channel] = cv2.warpPerspective(
            _SRGB_TO_LINEAR[sheet[..., channel]], homography, (width, height),
            flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

    gain = lighting_gain(base, mask, sigma=max(9.0, min(height, width) / 18.0))
    lit *= gain[..., None] ** strength
    colour, highlights = _surface_light(base, mask, surface_finish)
    if surface_finish != 'matte':
        reflection = cv2.resize(highlights, (width, height), interpolation=cv2.INTER_LINEAR)
        if reflectance is not None:
            coverage = cv2.warpPerspective(reflectance, homography, (width, height),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            reflection *= coverage.astype(np.float32) / 255.0
    for channel in range(3):
        tint = cv2.resize(colour[..., channel], (width, height), interpolation=cv2.INTER_LINEAR)
        lit[..., channel] *= tint ** strength
        if surface_finish != 'matte':
            lit[..., channel] += reflection * tint * strength
    # Near-white linear highlights roll off without dimming ordinary materials.
    lit = soft_clip(lit * 255.0, knee=0.95) / 255.0
    alpha = mask.astype(np.float32) / 255.0
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather,
                                 borderType=cv2.BORDER_CONSTANT)
    alpha[mask == 0] = 0
    alpha = alpha[..., None]
    lit *= alpha
    lit += _SRGB_TO_LINEAR[base] * (1 - alpha)
    dark = lit <= 0.0031308
    dark_values = lit[dark] * 12.92
    np.power(lit, 1 / 2.4, out=lit)
    lit *= 1.055
    lit -= 0.055
    lit[dark] = dark_values
    np.rint(lit * 255.0, out=lit)
    patch = np.clip(lit, 0, 255).astype(np.uint8)
    selected = mask > 0
    out[y:y + height, x:x + width][selected] = patch[selected]
    return out

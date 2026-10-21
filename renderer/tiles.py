"""Build a top-down image of a tiled floor at true physical scale."""
import cv2
import numpy as np


def tile_sheet_scale(faces: list[np.ndarray], tile_mm: tuple[float, float],
                     floor_mm: tuple[float, float], quad: list) -> float:
    """Match photo detail using source resolution and a roughly 16 MP cap."""
    points = np.asarray(quad, np.float64)
    horizontal = max(np.linalg.norm(points[1] - points[0]),
                     np.linalg.norm(points[2] - points[3])) / floor_mm[0]
    vertical = max(np.linalg.norm(points[2] - points[1]),
                   np.linalg.norm(points[3] - points[0])) / floor_mm[1]
    source = min(min(image.shape[1] / tile_mm[0], image.shape[0] / tile_mm[1])
                 for image in faces)
    # The builder needs at least two pixels per tile to preserve physical scale.
    minimum = 2 / min(tile_mm)
    budget = (16_000_000 / (floor_mm[0] * floor_mm[1])) ** 0.5
    return float(min(max(minimum, min(max(horizontal, vertical), source)), budget))


def paste(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Paste src into dst at (x, y), clipping whatever falls outside."""
    h_dst, w_dst = dst.shape[:2]
    h_src, w_src = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_dst, x + w_src), min(h_dst, y + h_src)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = src[y0 - y:y1 - y, x0 - x:x1 - x]


def build_tile_sheet(faces: list[np.ndarray], tile_mm: tuple[float, float],
                     floor_mm: tuple[float, float], px_per_mm: float,
                     layout: str = "grid", grout_mm: float = 2.0,
                     grout_bgr: tuple[int, int, int] = (205, 205, 205),
                     seed: int = 7) -> np.ndarray:
    """Top-down BGR image of the tiled floor, before any perspective.

    faces must be the distinct printed faces of ONE sku. Stone-look tiles ship
    4-8 of them; using only one makes the rendered floor visibly repeat.
    """
    tile_w_mm, tile_d_mm = tile_mm
    floor_w_mm, floor_d_mm = floor_mm
    tile_w = max(2, round(tile_w_mm * px_per_mm))
    tile_h = max(2, round(tile_d_mm * px_per_mm))
    gap = round(grout_mm * px_per_mm) if grout_mm > 0 else 0
    gap = max(0, min(gap, min(tile_w, tile_h) // 3))
    width = max(2, round(floor_w_mm * px_per_mm))
    height = max(2, round(floor_d_mm * px_per_mm))

    # Ground is grout colour; tiles are pasted inset by `gap`, so the gap IS
    # the grout line. Grout colour is a real buying decision, not a detail.
    sheet = np.full((height, width, 3), np.array(grout_bgr, np.uint8), np.uint8)
    resized = [cv2.resize(f, (tile_w - gap, tile_h - gap),
                          interpolation=cv2.INTER_AREA) for f in faces]

    # Stagger along the long axis of upright planks without rotating their grain.
    vertical_brick = layout == "brick" and tile_h > tile_w
    if vertical_brick:
        sheet = sheet.transpose(1, 0, 2)
        resized = [image.transpose(1, 0, 2) for image in resized]
        width, height = height, width
        tile_w, tile_h = tile_h, tile_w

    rng = np.random.default_rng(seed)
    row = y = 0
    while y < height:
        x = -tile_w // 2 if (layout == "brick" and row % 2) else 0
        while x < width:
            paste(sheet, resized[int(rng.integers(len(resized)))],
                  x + gap // 2, y + gap // 2)
            x += tile_w
        y += tile_h
        row += 1
    return sheet.transpose(1, 0, 2) if vertical_brick else sheet

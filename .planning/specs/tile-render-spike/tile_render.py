"""Spike: project a tile texture onto a floor in a photo, using plain geometry.

THROWAWAY. It exists to answer one question: does the render look convincing
enough that a customer believes it is their own floor? Not product code.

Four steps:
  1. floor region  -- 4 clicked points (exact), minus objects standing on it
  2. homography    -- those 4 points give the perspective matrix
  3. tile sheet    -- lay the texture at true physical scale, with grout and
                      face variation, then warp it through the homography
  4. relight       -- multiply by the low-frequency luminance of the original
"""
import argparse
import math
import sys

import cv2
import numpy as np


# ---------- step 3: build the tile sheet (top-down, mm -> px) ----------

def paste(dst, src, x, y):
    """Paste src into dst at (x, y), clipping whatever falls outside."""
    h_dst, w_dst = dst.shape[:2]
    h_src, w_src = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_dst, x + w_src), min(h_dst, y + h_src)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = src[y0 - y:y1 - y, x0 - x:x1 - x]


def build_tile_sheet(faces, tile_mm, floor_mm, px_per_mm, layout="grid",
                     grout_mm=2.0, grout_bgr=(205, 205, 205), seed=7):
    """Top-down image of the tiled floor, before any perspective is applied.

    faces    : the distinct printed faces of one SKU (stone-look tiles ship 4-8)
    tile_mm  : (width, depth) of one tile, e.g. (600, 600)
    floor_mm : (width, depth) of the floor
    layout   : "grid" (aligned) or "brick" (offset rows)
    """
    tile_w_mm, tile_d_mm = tile_mm
    floor_w_mm, floor_d_mm = floor_mm
    tile_w = max(2, round(tile_w_mm * px_per_mm))
    tile_h = max(2, round(tile_d_mm * px_per_mm))
    gap = round(grout_mm * px_per_mm) if grout_mm > 0 else 0
    gap = max(0, min(gap, min(tile_w, tile_h) // 3))
    width = max(2, round(floor_w_mm * px_per_mm))
    height = max(2, round(floor_d_mm * px_per_mm))

    # Fill with the grout colour, then leave a `gap` between tiles: the gap IS
    # the grout line. Grout colour is a real buying decision, not a detail.
    sheet = np.full((height, width, 3), np.array(grout_bgr, np.uint8), np.uint8)
    resized = [cv2.resize(f, (tile_w - gap, tile_h - gap), interpolation=cv2.INTER_AREA)
               for f in faces]

    rng = np.random.default_rng(seed)   # fixed seed: reopening gives the same floor
    row = y = 0
    while y < height:
        x = -tile_w // 2 if (layout == "brick" and row % 2) else 0
        while x < width:
            paste(sheet, resized[int(rng.integers(len(resized)))], x + gap // 2, y + gap // 2)
            x += tile_w
        y += tile_h
        row += 1
    return sheet


# ---------- step 2: homography ----------

def sheet_to_photo_homography(sheet_shape, quad):
    """quad order: FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT."""
    height, width = sheet_shape[:2]
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    return cv2.getPerspectiveTransform(src, np.float32(quad))


# ---------- step 1: floor region, minus whatever stands on it ----------

def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def near_edge_band(quad, quad_mask, depth=0.14):
    """A strip along the NEAR edge. Assumption: this is certainly bare floor."""
    height, width = quad_mask.shape[:2]
    band = np.int32([quad[3], quad[2],
                     lerp(quad[2], quad[1], depth), lerp(quad[3], quad[0], depth)])
    strip = np.zeros((height, width), np.uint8)
    cv2.fillPoly(strip, [band], 255)
    return cv2.bitwise_and(strip, quad_mask)


def drop_small_components(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def floor_mask_by_colour_deviation(base_bgr, quad, quad_mask, threshold=13.0,
                                   lightness_weight=0.15, min_object_area=400):
    """One-class model: learn the floor colour from the near-edge strip, then
    mark pixels that DEVIATE from it as objects standing on the floor.

    Lightness is weighted right down because a cast shadow moves L a long way
    while leaving hue alone. Measured on the test scene: lit floor d2=0.3,
    shadowed floor d2=0.5 -- so shadows stay classified as floor.

    Morphology runs on the OBJECTS, never on the floor. Closing the floor mask
    lets a few floor-looking pixels inside an object dilate outward and swallow
    the whole object -- measured at 12% -> 91% on the cabinet.
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
    objects = cv2.morphologyEx(objects, cv2.MORPH_CLOSE, kernel)   # solidify interiors
    objects = drop_small_components(objects, min_object_area)       # drop noise specks
    return cv2.bitwise_and(quad_mask, cv2.bitwise_not(objects))


def floor_mask_by_grabcut(base_bgr, quad, quad_mask, iters=4):
    """Kept only so the failure stays reproducible -- see README.

    GrabCut needs two colour models. Its background seed is everything outside
    the quad, which here is WALL -- the same grey as the floor. The two models
    collapse onto each other and it discriminates nothing.
    """
    height, width = quad_mask.shape[:2]
    gc = np.full((height, width), cv2.GC_BGD, np.uint8)
    gc[quad_mask > 0] = cv2.GC_PR_FGD
    gc[near_edge_band(quad, quad_mask) > 0] = cv2.GC_FGD
    cv2.grabCut(base_bgr, gc, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                iters, cv2.GC_INIT_WITH_MASK)
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def build_mask(base_bgr, quad, exclude=None, auto="none", threshold=13.0):
    """quad, minus hand-drawn exclusions, minus whatever `auto` detects.

    Hand-drawn is exact and always works; auto is a convenience that can be
    wrong. That ordering is deliberate -- never ship auto without the manual
    override underneath it.
    """
    height, width = base_bgr.shape[:2]
    quad_mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(quad_mask, np.int32(quad), 255)

    mask = quad_mask.copy()
    for poly in (exclude or []):
        cv2.fillPoly(mask, [np.int32(poly)], 0)

    if auto == "grabcut":
        mask = cv2.bitwise_and(mask, floor_mask_by_grabcut(base_bgr, quad, quad_mask))
    elif auto == "colour-deviation":
        mask = cv2.bitwise_and(mask, floor_mask_by_colour_deviation(
            base_bgr, quad, quad_mask, threshold=threshold))
    return mask


# ---------- step 4: relight ----------

def lighting_gain(base_bgr, mask, sigma=41.0, lo=0.55, hi=1.35):
    """Low-frequency luminance of the original, normalised around 1.0.

    This is what carries the room's real shadows onto the new floor. Blur hard
    so the OLD floor's texture cannot leak through.
    """
    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), sigma)
    selected = mask > 127
    if not selected.any():
        return np.ones_like(low)
    return np.clip(low / max(float(low[selected].mean()), 1e-6), lo, hi)


def soft_clip(v, knee=0.80):
    """Roll off the highlights. Hard-clipping at 255 flattens the veining away."""
    x = v / 255.0
    over = x > knee
    x[over] = knee + (1 - knee) * (1 - np.exp(-(x[over] - knee) / (1 - knee)))
    return np.clip(x * 255.0, 0, 255)


def render(base_bgr, sheet, quad, mask=None, strength=1.0, feather=2.0):
    height, width = base_bgr.shape[:2]
    warped = cv2.warpPerspective(sheet, sheet_to_photo_homography(sheet.shape, quad),
                                 (width, height), flags=cv2.INTER_AREA,
                                 borderMode=cv2.BORDER_REPLICATE)
    if mask is None:
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, np.int32(quad), 255)

    gain = lighting_gain(base_bgr, mask, sigma=max(9.0, min(height, width) / 18.0))
    lit = soft_clip(warped.astype(np.float32) * (gain[..., None] ** strength))
    alpha = (cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather) / 255.0)[..., None]
    out = base_bgr.astype(np.float32) * (1 - alpha) + lit * alpha
    return out.astype(np.uint8), mask


# ---------- the same measurements also price the job ----------

def estimate_quantity(floor_mm, tile_mm, waste_pct=8.0, tiles_per_box=4):
    floor_w, floor_d = floor_mm
    tile_w, tile_d = tile_mm
    area = (floor_w / 1000) * (floor_d / 1000)
    per_tile = (tile_w / 1000) * (tile_d / 1000)
    needed = math.ceil(area / per_tile * (1 + waste_pct / 100))
    return {
        "area_m2": round(area, 2),
        "tiles_exact": math.ceil(area / per_tile),
        "waste_pct": waste_pct,
        "tiles_to_buy": needed,
        "boxes": math.ceil(needed / tiles_per_box),
    }


# ---------- CLI ----------

def parse_points(text):
    return [tuple(float(v) for v in p.split(",")) for p in text.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--faces", required=True, help="texture images, comma separated")
    ap.add_argument("--quad", required=True, help="FAR-LEFT FAR-RIGHT NEAR-RIGHT NEAR-LEFT")
    ap.add_argument("--tile-mm", default="600x600")
    ap.add_argument("--floor-mm", default="4000x3000")
    ap.add_argument("--layout", default="grid", choices=["grid", "brick"])
    ap.add_argument("--grout-mm", type=float, default=2.0)
    ap.add_argument("--grout", default="205,205,205", help="grout colour, BGR")
    ap.add_argument("--exclude", default="", help="polygons to cut out, ';' separated")
    ap.add_argument("--auto-mask", default="none",
                    choices=["none", "grabcut", "colour-deviation"])
    ap.add_argument("--threshold", type=float, default=13.0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--ppm", type=float, default=0.35, help="px per mm for the tile sheet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = cv2.imread(args.photo, cv2.IMREAD_COLOR)
    if base is None:
        sys.exit(f"cannot read photo: {args.photo}")
    faces = []
    for path in args.faces.split(","):
        face = cv2.imread(path.strip(), cv2.IMREAD_COLOR)
        if face is None:
            sys.exit(f"cannot read texture: {path}")
        faces.append(face)

    tile_mm = tuple(float(v) for v in args.tile_mm.lower().split("x"))
    floor_mm = tuple(float(v) for v in args.floor_mm.lower().split("x"))
    quad = parse_points(args.quad)
    if len(quad) != 4:
        sys.exit("--quad needs exactly 4 points")
    exclude = [parse_points(p) for p in args.exclude.split(";") if p.strip()]

    sheet = build_tile_sheet(faces, tile_mm, floor_mm, args.ppm, layout=args.layout,
                             grout_mm=args.grout_mm,
                             grout_bgr=tuple(int(v) for v in args.grout.split(",")))
    mask = build_mask(base, quad, exclude=exclude, auto=args.auto_mask,
                      threshold=args.threshold)
    out, mask = render(base, sheet, quad, mask=mask, strength=args.strength)

    cv2.imwrite(args.out, out)
    cv2.imwrite(args.out.replace(".png", "_mask.png"), mask)
    cv2.imwrite(args.out.replace(".png", "_before-after.png"), np.hstack([base, out]))
    print(f"wrote {args.out}")
    print("quote from the SAME measurements:", estimate_quantity(floor_mm, tile_mm))


if __name__ == "__main__":
    main()

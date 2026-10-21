# Render Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local tool where the operator drops in a real room photo, clicks the four floor corners, picks a texture, and sees the render — so risk #1 (is the render convincing?) can be answered without anyone typing pixel coordinates.

**Architecture:** Harden the proven spike code into a tested Python package (`renderer/`), expose one render endpoint over FastAPI, and serve a single static page that does the clicking. Rendering happens server-side here on purpose: this bench validates *realism*, not interaction speed. The spec's client-side WebGL swap is a later plan and does not block this one.

**Tech Stack:** Python 3.14, OpenCV, NumPy, FastAPI, Uvicorn, pytest. Plain HTML + Canvas 2D on the client — no framework, no build step.

**Spec:** `.planning/specs/tile-visualizer-design.md`

## Global Constraints

- Python 3.14 (verified: 3.14.6 at `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`)
- OpenCV `cv2.__version__` 5.0.0, NumPy 2.5.2, FastAPI 0.138.2, Uvicorn 0.49.0 — all verified present on this machine
- All identifiers, comments, CLI flags and filenames in **English**. Conversation is Vietnamese; source is not.
- No GPU, no torch, no model downloads. Spec §4 measured that v1 does not need them.
- Quad point order is always **FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT**. Every function that takes a quad assumes this.
- Physical scale is never inferred from the image. Room dimensions are always supplied by the caller (spec §3).
- Exclusion regions are free-form polygons, never rectangles (spec §5).
- The spike at `.planning/specs/tile-render-spike/` is throwaway reference material. Copy logic from it deliberately; do not import from it.

---

### Task 1: Project scaffold and tile sheet builder

**Files:**
- Create: `requirements.txt`
- Create: `renderer/__init__.py`
- Create: `renderer/tiles.py`
- Test: `tests/test_tiles.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `build_tile_sheet(faces: list[np.ndarray], tile_mm: tuple[float, float], floor_mm: tuple[float, float], px_per_mm: float, layout: str = "grid", grout_mm: float = 2.0, grout_bgr: tuple[int, int, int] = (205, 205, 205), seed: int = 7) -> np.ndarray` — returns a top-down BGR image of the tiled floor
  - `paste(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None` — clipped paste, mutates `dst`

- [ ] **Step 1: Create the virtualenv and requirements**

```bash
cd "$(git rev-parse --show-toplevel)"
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m venv .venv
cat > requirements.txt <<'EOF'
opencv-python>=5.0,<6
numpy>=2.5,<3
fastapi>=0.138,<1
python-multipart>=0.0.9
uvicorn>=0.49,<1
pytest>=8
httpx>=0.27
EOF
.venv/bin/pip install -q -r requirements.txt
printf '.venv/\n__pycache__/\n*.pyc\n' > .gitignore
```

`pytest.ini` is required, not optional. With pytest's default `prepend` import
mode and no `__init__.py` under `tests/`, pytest inserts `tests/` into
`sys.path` — not the repo root. Both `import renderer` and
`from tests.fixtures import ...` would raise `ModuleNotFoundError` in every
task. `pythonpath` puts the repo root on the path and fixes both.

```bash
cat > pytest.ini <<'EOF'
[pytest]
pythonpath = .
testpaths = tests
EOF
```

- [x] **Step 2: Write the failing test**

```python
# tests/test_tiles.py
import numpy as np
import pytest
from renderer.tiles import build_tile_sheet


def face(colour):
    return np.full((600, 600, 3), colour, np.uint8)


def test_sheet_pixel_size_follows_physical_size():
    sheet = build_tile_sheet([face(200)], (600, 600), (4200, 3200), 0.5)
    assert sheet.shape[:2] == (1600, 2100)   # 3200mm*0.5, 4200mm*0.5


def test_tile_count_across_width_matches_floor_width():
    ppm = 0.5
    sheet = build_tile_sheet([face(200)], (600, 600), (3000, 3000), ppm,
                             grout_mm=0)
    tile_px = int(600 * ppm)
    assert sheet.shape[1] // tile_px == 5      # 3000mm / 600mm


def test_grout_colour_survives_between_tiles():
    grout = (10, 20, 30)
    sheet = build_tile_sheet([face(240)], (600, 600), (1800, 1800), 0.5,
                             grout_mm=10, grout_bgr=grout)
    assert (sheet == np.array(grout, np.uint8)).all(axis=2).any()


def test_same_seed_gives_identical_sheet():
    faces = [face(100), face(150), face(200), face(250)]
    a = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, seed=7)
    b = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, seed=7)
    assert np.array_equal(a, b)


def test_different_seed_changes_face_placement():
    faces = [face(100), face(150), face(200), face(250)]
    a = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, seed=7)
    b = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, seed=99)
    assert not np.array_equal(a, b)


def test_brick_layout_differs_from_grid():
    faces = [face(100), face(250)]
    grid = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, layout="grid", seed=3)
    brick = build_tile_sheet(faces, (600, 600), (3000, 3000), 0.4, layout="brick", seed=3)
    assert not np.array_equal(grid, brick)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_tiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'renderer'`

- [x] **Step 4: Write minimal implementation**

```python
# renderer/__init__.py
"""Tile render core. Proven in .planning/specs/tile-render-spike/, hardened here."""
```

```python
# renderer/tiles.py
"""Build a top-down image of a tiled floor at true physical scale."""
import cv2
import numpy as np


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
    return sheet
```

- [x] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tiles.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini .gitignore renderer/ tests/test_tiles.py
git commit -m "feat(renderer): tile sheet builder at true physical scale"
```

---

### Task 2: Geometry — homography and the near-edge band

**Files:**
- Create: `renderer/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Quad = list[tuple[float, float]]` — always FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT
  - `sheet_to_photo_homography(sheet_shape: tuple, quad: Quad) -> np.ndarray` — 3x3 float64
  - `lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]`
  - `near_edge_band(quad: Quad, quad_mask: np.ndarray, depth: float = 0.14) -> np.ndarray` — uint8 mask, 0 or 255
  - `quad_mask_of(shape: tuple, quad: Quad) -> np.ndarray` — uint8 mask of the filled quad

- [x] **Step 1: Write the failing test**

```python
# tests/test_geometry.py
import cv2
import numpy as np
from renderer.geometry import (lerp, near_edge_band, quad_mask_of,
                               sheet_to_photo_homography)

QUAD = [(335, 432), (952, 432), (1245, 828), (58, 828)]
SHAPE = (860, 1280)


def test_homography_maps_sheet_corners_onto_quad():
    sheet_shape = (400, 600, 3)
    h = sheet_to_photo_homography(sheet_shape, QUAD)
    corners = np.float32([[0, 0], [600, 0], [600, 400], [0, 400]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, h).reshape(-1, 2)
    assert np.allclose(mapped, np.float32(QUAD), atol=1e-3)


def test_lerp_endpoints_and_midpoint():
    assert lerp((0, 0), (10, 20), 0.0) == (0, 0)
    assert lerp((0, 0), (10, 20), 1.0) == (10, 20)
    assert lerp((0, 0), (10, 20), 0.5) == (5, 10)


def test_quad_mask_is_binary_and_covers_interior():
    m = quad_mask_of(SHAPE, QUAD)
    assert set(np.unique(m)).issubset({0, 255})
    assert m[800, 650] == 255      # well inside
    assert m[100, 650] == 0        # above the far edge


def test_near_edge_band_sits_inside_quad_and_near_the_bottom():
    qm = quad_mask_of(SHAPE, QUAD)
    band = near_edge_band(QUAD, qm)
    assert band.any()
    assert not (band[qm == 0] > 0).any()          # never escapes the quad
    ys = np.where(band > 0)[0]
    assert ys.min() > 750                          # only near the NEAR edge


def test_deeper_band_covers_more_area():
    qm = quad_mask_of(SHAPE, QUAD)
    shallow = (near_edge_band(QUAD, qm, depth=0.10) > 0).sum()
    deep = (near_edge_band(QUAD, qm, depth=0.30) > 0).sum()
    assert deep > shallow
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'renderer.geometry'`

- [x] **Step 3: Write minimal implementation**

```python
# renderer/geometry.py
"""Perspective mapping between the top-down tile sheet and the photo.

Quad order is FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT everywhere.
"""
import cv2
import numpy as np


def quad_mask_of(shape, quad):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.int32(quad), 255)
    return mask


def sheet_to_photo_homography(sheet_shape, quad):
    height, width = sheet_shape[:2]
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    return cv2.getPerspectiveTransform(src, np.float32(quad))


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def near_edge_band(quad, quad_mask, depth=0.14):
    """Strip along the NEAR edge. Assumption: this is certainly bare floor.

    Used as the seed for the colour model in renderer.masking.
    """
    band = np.int32([quad[3], quad[2],
                     lerp(quad[2], quad[1], depth), lerp(quad[3], quad[0], depth)])
    strip = np.zeros(quad_mask.shape[:2], np.uint8)
    cv2.fillPoly(strip, [band], 255)
    return cv2.bitwise_and(strip, quad_mask)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_geometry.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add renderer/geometry.py tests/test_geometry.py
git commit -m "feat(renderer): homography and near-edge band"
```

---

### Task 3: Masking — hand-drawn exclusions and the colour-deviation convenience

**Files:**
- Create: `renderer/masking.py`
- Create: `tests/fixtures.py`
- Test: `tests/test_masking.py`

**Interfaces:**
- Consumes: `renderer.geometry.quad_mask_of`, `near_edge_band`
- Produces:
  - `build_mask(base_bgr: np.ndarray, quad: Quad, exclude: list[list[tuple[float, float]]] | None = None, auto: str = "none", threshold: float = 13.0) -> np.ndarray` — uint8 floor mask; `auto` is `"none"` or `"colour-deviation"`
  - `floor_mask_by_colour_deviation(base_bgr, quad, quad_mask, threshold=13.0, lightness_weight=0.15, min_object_area=400) -> np.ndarray`
  - `drop_small_components(mask: np.ndarray, min_area: int) -> np.ndarray`

- [x] **Step 1: Write the shared fixture and the failing test**

```python
# tests/fixtures.py
"""Synthetic room used by the masking and compose tests.

A grey floor in perspective, a strong window gradient, a soft cast shadow, a
brown cabinet (chromatic — colour masking should catch it) and a near-grey pot
(colour masking provably cannot catch it; see spec section 5).
"""
import cv2
import numpy as np

W, H = 1280, 860
QUAD = [(335, 432), (952, 432), (1245, 828), (58, 828)]
CABINET_FRONT = (slice(600, 686), slice(830, 1030))
POT_BODY = (slice(600, 666), slice(210, 290))
BARE_FLOOR = (slice(740, 810), slice(450, 700))


def room_with_objects():
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        t = y / H
        img[y, :] = (int(196 - 26 * t), int(198 - 26 * t), int(201 - 26 * t))

    rng = np.random.default_rng(3)
    floor = np.full((H, W, 3), (138, 141, 144), np.uint8)
    floor = np.clip(floor.astype(np.int16) + rng.normal(0, 7, floor.shape),
                    0, 255).astype(np.uint8)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(QUAD), 255)
    img[mask > 0] = floor[mask > 0]

    gx = np.linspace(1.14, 0.88, W, dtype=np.float32)[None, :]
    gy = np.linspace(1.08, 0.94, H, dtype=np.float32)[:, None]
    img = np.clip(img.astype(np.float32) * (gx * gy)[..., None], 0, 255).astype(np.uint8)

    shade = np.zeros((H, W), np.float32)
    cv2.ellipse(shade, (1010, 770), (250, 100), 12, 0, 360, 1.0, -1)
    shade = cv2.GaussianBlur(shade, (0, 0), 46)
    img = np.clip(img.astype(np.float32) * (1 - 0.36 * shade[..., None]),
                  0, 255).astype(np.uint8)

    base = [(800, 596), (1000, 596), (1040, 688), (820, 688)]
    top = [(x, y - 178) for x, y in base]
    cv2.fillPoly(img, [np.int32([base[3], base[2], top[2], top[3]])], (66, 92, 122))
    cv2.fillPoly(img, [np.int32([base[2], base[1], top[1], top[2]])], (48, 68, 92))
    cv2.fillPoly(img, [np.int32(top)], (86, 116, 150))

    for a, b in ((0, 1), (0, 3), (1, 2)):          # skirting
        cv2.line(img, QUAD[a], QUAD[b], (168, 170, 172), 3)

    contact = np.zeros((H, W), np.float32)         # contact shadows
    cv2.ellipse(contact, (920, 694), (140, 26), 0, 0, 360, 1.0, -1)
    cv2.ellipse(contact, (250, 648), (74, 22), 0, 0, 360, 1.0, -1)
    contact = cv2.GaussianBlur(contact, (0, 0), 15)
    img = np.clip(img.astype(np.float32) * (1 - 0.45 * contact[..., None]),
                  0, 255).astype(np.uint8)

    # Near-grey on purpose: this is the object colour masking provably cannot
    # separate from a grey floor (spec section 5).
    cv2.ellipse(img, (250, 648), (58, 20), 0, 0, 360, (92, 106, 126), -1)
    cv2.rectangle(img, (206, 574), (294, 648), (88, 100, 120), -1)
    leaf_rng = np.random.default_rng(9)
    for _ in range(90):
        angle, radius = leaf_rng.uniform(0, 2 * np.pi), leaf_rng.uniform(10, 96)
        cv2.circle(img, (int(250 + np.cos(angle) * radius),
                         int(648 - 96 - np.sin(angle) * radius * 0.75)),
                   int(leaf_rng.integers(7, 17)), (58, 108, 62), -1)

    # Camera noise matters: it sets the colour spread the one-class model
    # normalises by, so the measured thresholds only hold with it present.
    return np.clip(img.astype(np.float32) + rng.normal(0, 3.2, img.shape),
                   0, 255).astype(np.uint8)


def kept_pct(mask, region):
    """Percent of `region` that the mask EXCLUDES from the floor."""
    return 100.0 - 100.0 * (mask[region] > 127).mean()


def tiled_pct(mask, region):
    """Percent of `region` the mask treats as floor (and so will tile over)."""
    return 100.0 * (mask[region] > 127).mean()
```

```python
# tests/test_masking.py
import numpy as np
from renderer.masking import build_mask, drop_small_components
from tests.fixtures import (BARE_FLOOR, CABINET_FRONT, POT_BODY, QUAD,
                            kept_pct, room_with_objects, tiled_pct)


def test_plain_quad_covers_the_whole_floor_and_nothing_above_it():
    base = room_with_objects()
    m = build_mask(base, QUAD)
    assert tiled_pct(m, BARE_FLOOR) == 100.0
    assert m[100, 650] == 0


def test_hand_drawn_exclusion_removes_exactly_that_polygon():
    base = room_with_objects()
    poly = [(820, 500), (1045, 500), (1045, 694), (820, 694)]
    m = build_mask(base, QUAD, exclude=[poly])
    assert kept_pct(m, CABINET_FRONT) == 100.0
    assert tiled_pct(m, BARE_FLOOR) == 100.0     # untouched elsewhere


def test_colour_deviation_catches_the_chromatic_cabinet():
    base = room_with_objects()
    m = build_mask(base, QUAD, auto="colour-deviation", threshold=13.0)
    assert kept_pct(m, CABINET_FRONT) > 90.0


def test_colour_deviation_leaves_shadowed_floor_alone():
    """Lightness is weighted down so a cast shadow is not read as an object."""
    base = room_with_objects()
    m = build_mask(base, QUAD, auto="colour-deviation", threshold=13.0)
    assert tiled_pct(m, BARE_FLOOR) > 99.0


def test_morphology_must_not_swallow_objects():
    """Regression. Closing the FLOOR mask let stray floor-looking pixels inside
    the cabinet dilate outward and merge, taking the cabinet from 88% correctly
    excluded to 9%. Morphology must act on the objects instead."""
    base = room_with_objects()
    m = build_mask(base, QUAD, auto="colour-deviation", threshold=13.0)
    assert kept_pct(m, CABINET_FRONT) > 90.0


def test_near_neutral_object_is_documented_as_out_of_reach():
    """Spec section 5: no threshold keeps a near-grey object AND the floor.
    This test pins that limitation so nobody 'fixes' it by lowering the
    threshold and silently shredding the floor."""
    base = room_with_objects()
    usable = build_mask(base, QUAD, auto="colour-deviation", threshold=13.0)
    aggressive = build_mask(base, QUAD, auto="colour-deviation", threshold=6.0)
    assert kept_pct(usable, POT_BODY) < 50.0          # missed, as expected
    assert kept_pct(aggressive, POT_BODY) > 90.0      # caught, but...
    assert tiled_pct(aggressive, BARE_FLOOR) < 60.0   # ...floor is destroyed


def test_drop_small_components_removes_specks_and_keeps_blobs():
    m = np.zeros((100, 100), np.uint8)
    m[10:14, 10:14] = 255          # 16 px speck
    m[50:80, 50:80] = 255          # 900 px blob
    out = drop_small_components(m, 400)
    assert out[12, 12] == 0
    assert out[60, 60] == 255
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_masking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'renderer.masking'`

- [x] **Step 3: Write minimal implementation**

```python
# renderer/masking.py
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
               auto: str = "none", threshold: float = 13.0) -> np.ndarray:
    """Floor quad, minus hand-drawn polygons, minus whatever `auto` detects."""
    quad_mask = quad_mask_of(base_bgr.shape, quad)
    mask = quad_mask.copy()
    for poly in (exclude or []):
        cv2.fillPoly(mask, [np.int32(poly)], 0)

    if auto == "colour-deviation":
        mask = cv2.bitwise_and(mask, floor_mask_by_colour_deviation(
            base_bgr, quad, quad_mask, threshold=threshold))
    elif auto != "none":
        raise ValueError(f"unknown auto mask mode: {auto!r}")
    return mask
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_masking.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add renderer/masking.py tests/fixtures.py tests/test_masking.py
git commit -m "feat(renderer): floor masking with hand-drawn exclusions"
```

---

### Task 4: Compose — relight and blend

**Files:**
- Create: `renderer/compose.py`
- Test: `tests/test_compose.py`

**Interfaces:**
- Consumes: `renderer.geometry.sheet_to_photo_homography`, `renderer.geometry.quad_mask_of`
- Produces:
  - `soft_clip(v: np.ndarray, knee: float = 0.80) -> np.ndarray` — float array in [0, 255]
  - `lighting_gain(base_bgr, mask, sigma: float = 41.0, lo: float = 0.55, hi: float = 1.35) -> np.ndarray`
  - `render(base_bgr, sheet, quad, mask=None, strength: float = 1.0, feather: float = 2.0) -> np.ndarray` — returns the composited BGR image only

- [x] **Step 1: Write the failing test**

```python
# tests/test_compose.py
import numpy as np
from renderer.compose import lighting_gain, render, soft_clip
from renderer.geometry import quad_mask_of
from renderer.masking import build_mask
from renderer.tiles import build_tile_sheet
from tests.fixtures import BARE_FLOOR, QUAD, room_with_objects


def test_soft_clip_never_exceeds_255():
    v = np.linspace(0, 900, 500, dtype=np.float32)
    assert soft_clip(v.copy()).max() <= 255.0


def test_soft_clip_is_monotonic():
    v = np.linspace(0, 900, 500, dtype=np.float32)
    out = soft_clip(v.copy())
    assert (np.diff(out) >= -1e-4).all()


def test_soft_clip_leaves_midtones_alone():
    v = np.array([0.0, 100.0, 150.0], np.float32)
    assert np.allclose(soft_clip(v.copy()), v, atol=1e-3)


def test_soft_clip_compresses_above_the_knee_rather_than_clamping():
    """The discriminating test. Every other soft_clip test here also passes
    against a plain np.clip(v, 0, 255), so without this one the roll-off has
    no regression cover at all -- and losing the roll-off returns the render
    to flat blown highlights with the tile veining erased.
    """
    # Just above the knee (0.80 * 255 = 204): compressed, but not clamped.
    above_knee = float(soft_clip(np.array([220.0], np.float32))[0])
    assert 204.0 < above_knee < 220.0

    # Far above white: a clamp yields exactly 255; the roll-off asymptotes
    # toward it and must stay strictly below.
    far = float(soft_clip(np.array([400.0], np.float32))[0])
    assert far < 254.9
    assert far > above_knee


def test_lighting_gain_is_centred_on_one_inside_the_mask():
    base = room_with_objects()
    mask = quad_mask_of(base.shape, QUAD)
    gain = lighting_gain(base, mask)
    assert 0.85 < float(gain[mask > 127].mean()) < 1.15


def test_lighting_gain_is_darker_where_the_shadow_is():
    base = room_with_objects()
    mask = quad_mask_of(base.shape, QUAD)
    gain = lighting_gain(base, mask)
    assert gain[770, 1010] < gain[770, 300]     # shadow vs window side


def test_render_leaves_pixels_outside_the_mask_untouched():
    base = room_with_objects()
    sheet = build_tile_sheet([np.full((600, 600, 3), 235, np.uint8)],
                             (600, 600), (4200, 3200), 0.35)
    mask = build_mask(base, QUAD)
    out = render(base, sheet, QUAD, mask=mask)
    assert np.array_equal(out[100, :, :], base[100, :, :])   # above far edge


def test_render_changes_the_floor():
    base = room_with_objects()
    sheet = build_tile_sheet([np.full((600, 600, 3), 235, np.uint8)],
                             (600, 600), (4200, 3200), 0.35)
    out = render(base, sheet, QUAD, mask=build_mask(base, QUAD))
    assert not np.array_equal(out[BARE_FLOOR], base[BARE_FLOOR])
    assert out[BARE_FLOOR].mean() > base[BARE_FLOOR].mean()   # cream over grey


def test_excluded_region_is_not_painted_over():
    base = room_with_objects()
    sheet = build_tile_sheet([np.full((600, 600, 3), 235, np.uint8)],
                             (600, 600), (4200, 3200), 0.35)
    poly = [(820, 500), (1045, 500), (1045, 694), (820, 694)]
    out = render(base, sheet, QUAD, mask=build_mask(base, QUAD, exclude=[poly]))
    assert np.array_equal(out[620:680, 850:1010], base[620:680, 850:1010])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_compose.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'renderer.compose'`

- [x] **Step 3: Write minimal implementation**

```python
# renderer/compose.py
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
    alpha = (cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather) / 255.0)[..., None]
    out = base_bgr.astype(np.float32) * (1 - alpha) + lit * alpha
    return out.astype(np.uint8)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_compose.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add renderer/compose.py tests/test_compose.py
git commit -m "feat(renderer): relight and blend the warped tile sheet"
```

---

### Task 5: Texture registry — physical size is never guessed

**Files:**
- Create: `renderer/catalogue.py`
- Create: `textures/registry.json`
- Test: `tests/test_catalogue.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `@dataclass Sku(name: str, tile_mm: tuple[float, float], faces: list[str], tiles_per_box: int, finish: str)` with property `m2_per_tile: float`
  - `load_registry(path: str | Path) -> dict[str, Sku]`
  - `load_faces(sku: Sku, root: str | Path) -> list[np.ndarray]`

- [x] **Step 1: Write the failing test**

```python
# tests/test_catalogue.py
import json

import cv2
import numpy as np
import pytest
from renderer.catalogue import Sku, load_faces, load_registry


def write_registry(tmp_path, entries):
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(entries))
    return p


def test_loads_sku_with_physical_size(tmp_path):
    p = write_registry(tmp_path, {
        "marble-600": {"tile_mm": [600, 600], "faces": ["a.png", "b.png"],
                       "tiles_per_box": 4, "finish": "matte"}})
    reg = load_registry(p)
    assert reg["marble-600"].tile_mm == (600.0, 600.0)
    assert reg["marble-600"].tiles_per_box == 4
    assert reg["marble-600"].m2_per_tile == pytest.approx(0.36)


def test_missing_tile_mm_is_an_error_not_a_default(tmp_path):
    """Guessing physical size silently renders the floor at the wrong scale."""
    p = write_registry(tmp_path, {"bad": {"faces": ["a.png"], "tiles_per_box": 4,
                                     "finish": "matte"}})
    with pytest.raises(KeyError):
        load_registry(p)


def test_single_face_is_rejected_for_stone_look(tmp_path):
    """One face makes the rendered floor visibly repeat. Warn loudly."""
    p = write_registry(tmp_path, {"one": {"tile_mm": [600, 600], "faces": [],
                                     "tiles_per_box": 4, "finish": "matte"}})
    with pytest.raises(ValueError, match="at least one face"):
        load_registry(p)


def test_load_faces_reads_every_declared_file(tmp_path):
    tmp = tmp_path
    for name in ("a.png", "b.png"):
        cv2.imwrite(str(tmp / name), np.full((60, 60, 3), 200, np.uint8))
    sku = Sku("s", (600.0, 600.0), ["a.png", "b.png"], 4, "matte")
    faces = load_faces(sku, tmp)
    assert len(faces) == 2
    assert all(f.shape == (60, 60, 3) for f in faces)


def test_load_faces_fails_loudly_on_a_missing_file(tmp_path):
    tmp = tmp_path
    sku = Sku("s", (600.0, 600.0), ["nope.png"], 4, "matte")
    with pytest.raises(FileNotFoundError):
        load_faces(sku, tmp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_catalogue.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'renderer.catalogue'`

- [x] **Step 3: Write minimal implementation**

```python
# renderer/catalogue.py
"""Texture registry. Physical tile size is declared, never inferred."""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Sku:
    name: str
    tile_mm: tuple[float, float]
    faces: list[str]
    tiles_per_box: int
    finish: str

    @property
    def m2_per_tile(self) -> float:
        return (self.tile_mm[0] / 1000) * (self.tile_mm[1] / 1000)


def load_registry(path: str | Path) -> dict[str, Sku]:
    entries = json.loads(Path(path).read_text())
    registry = {}
    for name, e in entries.items():
        tile_mm = e["tile_mm"]            # KeyError on purpose: never default this
        faces = e.get("faces", [])
        if not faces:
            raise ValueError(f"sku {name!r} needs at least one face image; a "
                             "single face makes the rendered floor repeat")
        registry[name] = Sku(name, (float(tile_mm[0]), float(tile_mm[1])),
                             list(faces), int(e["tiles_per_box"]),
                             str(e.get("finish", "matte")))
    return registry


def load_faces(sku: Sku, root: str | Path) -> list[np.ndarray]:
    root = Path(root)
    images = []
    for rel in sku.faces:
        p = root / rel
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"texture face not readable: {p}")
        images.append(img)
    return images
```

`textures/registry.json` — starts as an empty object. Real SKUs land here once
the supplier texture question in spec section 7 is answered. It must stay valid
JSON, so the explanation lives here in the plan, not in the file.

```json
{}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_catalogue.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add renderer/catalogue.py textures/registry.json tests/test_catalogue.py
git commit -m "feat(renderer): texture registry with declared physical size"
```

---

### Task 6: The bench — FastAPI endpoint and the four-point picker

**Files:**
- Create: `bench/app.py`
- Create: `bench/static/index.html`
- Test: `tests/test_bench_api.py`

**Interfaces:**
- Consumes: everything from Tasks 1-5
- Produces: `POST /render` (multipart: `photo` file, `payload` JSON string) returning `image/png`; `GET /` returning the picker page

- [x] **Step 1: Write the failing test**

```python
# tests/test_bench_api.py
import io
import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from bench.app import app
from tests.fixtures import QUAD, room_with_objects

client = TestClient(app)


def photo_bytes():
    ok, buf = cv2.imencode(".png", room_with_objects())
    assert ok
    return io.BytesIO(buf.tobytes())


def face_bytes(value=235):
    ok, buf = cv2.imencode(".png", np.full((600, 600, 3), value, np.uint8))
    assert ok
    return io.BytesIO(buf.tobytes())


def payload(**over):
    body = {"quad": QUAD, "tile_mm": [600, 600], "floor_mm": [4200, 3200],
            "grout_mm": 2.0, "grout_bgr": [205, 205, 205], "layout": "grid",
            "auto": "none", "exclude": []}
    body.update(over)
    return json.dumps(body)


def test_index_page_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "canvas" in r.text.lower()


def test_render_returns_a_png_the_same_size_as_the_photo():
    r = client.post("/render",
                    files={"photo": ("room.png", photo_bytes(), "image/png"),
                           "faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload()})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == room_with_objects().shape


def test_render_actually_changes_the_floor():
    r = client.post("/render",
                    files={"photo": ("room.png", photo_bytes(), "image/png"),
                           "faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload()})
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    base = room_with_objects()
    assert out[770, 600].mean() > base[770, 600].mean()


def test_quad_with_wrong_point_count_is_rejected():
    r = client.post("/render",
                    files={"photo": ("room.png", photo_bytes(), "image/png"),
                           "faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload(quad=[[0, 0], [10, 0], [10, 10]])})
    assert r.status_code == 422


def test_unknown_auto_mode_is_rejected():
    r = client.post("/render",
                    files={"photo": ("room.png", photo_bytes(), "image/png"),
                           "faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload(auto="magic")})
    assert r.status_code == 422


def test_missing_photo_is_rejected():
    r = client.post("/render",
                    files={"faces": ("f1.png", face_bytes(), "image/png")},
                    data={"payload": payload()})
    assert r.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_bench_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench'`

- [x] **Step 3: Write the API**

```python
# bench/app.py
"""Validation bench. Not the product.

Rendering happens server-side here on purpose: this exists to answer whether
the render looks convincing on a real photo, not to be fast. The spec's
client-side WebGL swap is a separate plan.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from renderer.compose import render
from renderer.masking import build_mask
from renderer.tiles import build_tile_sheet

app = FastAPI(title="Render bench")
STATIC = Path(__file__).parent / "static"
VALID_AUTO = {"none", "colour-deviation"}


def _decode(raw: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "could not decode image")
    return img


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.post("/render")
async def do_render(payload: str = Form(...),
                    photo: UploadFile = File(...),
                    faces: list[UploadFile] = File(...)):
    try:
        body = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"payload is not JSON: {exc}")

    quad = body.get("quad") or []
    if len(quad) != 4:
        raise HTTPException(422, "quad needs exactly 4 points, ordered "
                                 "FAR-LEFT FAR-RIGHT NEAR-RIGHT NEAR-LEFT")
    auto = body.get("auto", "none")
    if auto not in VALID_AUTO:
        raise HTTPException(422, f"auto must be one of {sorted(VALID_AUTO)}")

    base = _decode(await photo.read())
    face_images = [_decode(await f.read()) for f in faces]
    if not face_images:
        raise HTTPException(422, "at least one texture face is required")

    sheet = build_tile_sheet(
        face_images,
        tuple(body.get("tile_mm", (600, 600))),
        tuple(body.get("floor_mm", (4000, 3000))),
        float(body.get("px_per_mm", 0.35)),
        layout=body.get("layout", "grid"),
        grout_mm=float(body.get("grout_mm", 2.0)),
        grout_bgr=tuple(body.get("grout_bgr", (205, 205, 205))))

    mask = build_mask(base, [tuple(p) for p in quad],
                      exclude=[[tuple(p) for p in poly]
                               for poly in body.get("exclude", [])],
                      auto=auto,
                      threshold=float(body.get("threshold", 13.0)))
    out = render(base, sheet, [tuple(p) for p in quad], mask=mask,
                 strength=float(body.get("strength", 1.0)))

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(500, "failed to encode result")
    return Response(buf.tobytes(), media_type="image/png")
```

- [x] **Step 4: Write the picker page**

```html
<!-- bench/static/index.html -->
<!doctype html>
<meta charset="utf-8">
<title>Render bench</title>
<style>
  body { font: 14px system-ui; margin: 16px; background:#111; color:#eee; }
  #wrap { position: relative; display: inline-block; }
  canvas { max-width: 100%; cursor: crosshair; border:1px solid #444; }
  .row { margin: 8px 0; }
  label { margin-right: 12px; }
  input[type=number] { width: 84px; }
  #result { max-width: 100%; border:1px solid #444; margin-top:12px; }
  .hint { color:#9ab; }
</style>

<h2>Render bench</h2>
<p class="hint">Click the four floor corners in this order:
  <b>far-left, far-right, near-right, near-left</b>. Shift-drag to paint an
  exclusion area over anything standing on the floor.</p>

<div class="row">
  <label>Photo <input type="file" id="photo" accept="image/*"></label>
  <label>Texture faces <input type="file" id="faces" accept="image/*" multiple></label>
</div>
<div class="row">
  <label>Tile mm <input type="number" id="tileW" value="600"> ×
    <input type="number" id="tileD" value="600"></label>
  <label>Floor mm <input type="number" id="floorW" value="4200"> ×
    <input type="number" id="floorD" value="3200"></label>
</div>
<div class="row">
  <label>Grout mm <input type="number" id="groutMm" value="2" step="0.5"></label>
  <label>Layout <select id="layout"><option>grid</option><option>brick</option></select></label>
  <label>Auto mask <select id="auto">
    <option value="none">none</option>
    <option value="colour-deviation">colour-deviation</option></select></label>
  <button id="go">Render</button>
  <button id="reset">Clear points</button>
</div>

<div id="wrap"><canvas id="cv"></canvas></div>
<img id="result" hidden>

<script>
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let img = null, quad = [], strokes = [], painting = null;

document.getElementById('photo').onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  const i = new Image();
  i.onload = () => { img = i; cv.width = i.width; cv.height = i.height;
                     quad = []; strokes = []; draw(); };
  i.src = URL.createObjectURL(f);
};

function pos(ev) {
  const r = cv.getBoundingClientRect();
  return [Math.round((ev.clientX - r.left) * cv.width / r.width),
          Math.round((ev.clientY - r.top) * cv.height / r.height)];
}

cv.onmousedown = ev => {
  if (!img) return;
  if (ev.shiftKey) { painting = [pos(ev)]; strokes.push(painting); }
  else if (quad.length < 4) { quad.push(pos(ev)); draw(); }
};
cv.onmousemove = ev => { if (painting) { painting.push(pos(ev)); draw(); } };
window.onmouseup = () => { painting = null; };

document.getElementById('reset').onclick = () => { quad = []; strokes = []; draw(); };

function draw() {
  if (!img) return;
  ctx.drawImage(img, 0, 0);
  ctx.lineWidth = 3; ctx.strokeStyle = '#0f0'; ctx.fillStyle = '#0f0';
  quad.forEach(([x, y], n) => {
    ctx.beginPath(); ctx.arc(x, y, 7, 0, 7); ctx.fill();
    ctx.fillText(String(n + 1), x + 10, y - 10);
  });
  if (quad.length === 4) {
    ctx.beginPath(); ctx.moveTo(...quad[0]);
    quad.slice(1).forEach(p => ctx.lineTo(...p));
    ctx.closePath(); ctx.stroke();
  }
  ctx.strokeStyle = 'rgba(255,80,80,.85)'; ctx.lineWidth = 26;
  ctx.lineJoin = ctx.lineCap = 'round';
  strokes.forEach(s => {
    ctx.beginPath(); ctx.moveTo(...s[0]);
    s.slice(1).forEach(p => ctx.lineTo(...p)); ctx.stroke();
  });
}

// A painted stroke becomes an exclusion polygon: the stroke path out, then
// back offset by the brush radius. Crude but it is a free-form region, which
// is what the spec requires instead of rectangles.
function strokeToPolygon(s, r = 13) {
  return s.map(([x, y]) => [x, y - r]).concat(s.slice().reverse().map(([x, y]) => [x, y + r]));
}

document.getElementById('go').onclick = async () => {
  if (quad.length !== 4) return alert('Click exactly 4 corners first.');
  const faces = document.getElementById('faces').files;
  if (!faces.length) return alert('Pick at least one texture face.');
  const num = id => Number(document.getElementById(id).value);

  const fd = new FormData();
  fd.append('photo', document.getElementById('photo').files[0]);
  for (const f of faces) fd.append('faces', f);
  fd.append('payload', JSON.stringify({
    quad,
    tile_mm: [num('tileW'), num('tileD')],
    floor_mm: [num('floorW'), num('floorD')],
    grout_mm: num('groutMm'),
    layout: document.getElementById('layout').value,
    auto: document.getElementById('auto').value,
    exclude: strokes.filter(s => s.length > 1).map(s => strokeToPolygon(s)),
  }));

  const r = await fetch('/render', { method: 'POST', body: fd });
  if (!r.ok) return alert('Render failed: ' + r.status + ' ' + await r.text());
  const out = document.getElementById('result');
  out.src = URL.createObjectURL(await r.blob());
  out.hidden = false;
};
</script>
```

- [x] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_bench_api.py -v`
Expected: 6 passed

- [x] **Step 6: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: 38 passed

- [x] **Step 7: Start the bench and confirm it loads**

```bash
.venv/bin/uvicorn bench.app:app --port 8100
```
Open `http://127.0.0.1:8100`. Load a photo, click four corners, pick a texture, press Render.

- [x] **Step 8: Commit**

```bash
git add bench/ tests/test_bench_api.py
git commit -m "feat(bench): four-point picker and render endpoint"
```

---

### Task 7: Quote engine — aggregate by SKU before rounding

**Files:**
- Create: `renderer/quote.py`
- Test: `tests/test_quote.py`

**Interfaces:**
- Consumes: `renderer.catalogue.Sku`
- Produces:
  - `@dataclass Room(name: str, kind: str, size_m: tuple[float, float], sku_name: str)` with property `area: float`
  - `boxes_for(area_m2: float, sku: Sku, waste_pct: float) -> tuple[int, int]` — `(boxes, tiles)`
  - `quote_by_sku(rooms: list[Room], skus: dict[str, Sku], waste_pct: float = 8.0) -> dict[str, dict]`
  - `quote_per_room(rooms, skus, waste_pct=8.0) -> dict[str, dict]` — kept only to demonstrate the difference

- [x] **Step 1: Write the failing test**

```python
# tests/test_quote.py
import pytest
from renderer.catalogue import Sku
from renderer.quote import Room, boxes_for, quote_by_sku, quote_per_room

GRANITE = Sku("granite-800", (800.0, 800.0), ["a.png"], 3, "polished")
WOOD = Sku("wood-look-600", (600.0, 600.0), ["a.png"], 4, "matte")
KITCHEN = Sku("kitchen-600", (600.0, 600.0), ["a.png"], 4, "matte")
WC = Sku("wc-300", (300.0, 300.0), ["a.png"], 11, "matte")
SKUS = {s.name: s for s in (GRANITE, WOOD, KITCHEN, WC)}

TOWNHOUSE = [
    Room("living", "living", (4.2, 3.8), "granite-800"),
    Room("bed 1", "bedroom", (3.6, 3.2), "wood-look-600"),
    Room("bed 2", "bedroom", (3.2, 3.0), "wood-look-600"),
    Room("bed 3", "bedroom", (3.0, 3.0), "wood-look-600"),
    Room("kitchen", "kitchen", (3.0, 3.5), "kitchen-600"),
    Room("wc 1", "wc", (1.8, 2.2), "wc-300"),
    Room("wc 2", "wc", (1.6, 2.0), "wc-300"),
]


def test_room_area():
    assert Room("r", "bedroom", (3.0, 4.0), "wood-look-600").area == pytest.approx(12.0)


def test_boxes_for_includes_waste_and_rounds_up():
    boxes, tiles = boxes_for(11.52, WOOD, 8.0)
    assert tiles == 35          # ceil(11.52/0.36 * 1.08)
    assert boxes == 9           # ceil(35/4)


def test_zero_waste_is_honoured():
    _, tiles = boxes_for(3.6, WOOD, 0.0)
    assert tiles == 10          # ceil(3.6/0.36)


def test_aggregating_by_sku_never_costs_more_than_per_room():
    by_sku = quote_by_sku(TOWNHOUSE, SKUS)
    per_room = quote_per_room(TOWNHOUSE, SKUS)
    for name in by_sku:
        assert by_sku[name]["boxes"] <= per_room[name]["boxes"]


def test_measured_townhouse_totals():
    """Spec section 6: 50 boxes per-room vs 48 aggregated by sku."""
    assert sum(v["boxes"] for v in quote_per_room(TOWNHOUSE, SKUS).values()) == 50
    assert sum(v["boxes"] for v in quote_by_sku(TOWNHOUSE, SKUS).values()) == 48


def test_by_sku_reports_which_rooms_share_a_sku():
    q = quote_by_sku(TOWNHOUSE, SKUS)
    assert sorted(q["wood-look-600"]["rooms"]) == ["bed 1", "bed 2", "bed 3"]
    assert q["wood-look-600"]["area_m2"] == pytest.approx(30.12)


def test_unknown_sku_name_is_an_error():
    with pytest.raises(KeyError):
        quote_by_sku([Room("x", "bedroom", (3.0, 3.0), "nope")], SKUS)
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_quote.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'renderer.quote'`

- [x] **Step 3: Write minimal implementation**

```python
# renderer/quote.py
"""Quantities and boxes for a whole-house purchase.

Customers here buy for every room at once, so the same sku lands in several
rooms. Rounding boxes per room wastes a partial box each time; rounding once
per sku wastes it once. The measured saving is only ~4%, so do NOT sell on it.
The real reason is batch consistency: one order line for one sku makes same-lot
fulfilment natural, while three per-room lines can be filled from three
production lots -- and three bedrooms on "the same" tile then arrive in three
different shades.
"""
import math
from dataclasses import dataclass

from renderer.catalogue import Sku


@dataclass
class Room:
    name: str
    kind: str                     # living | bedroom | wc | kitchen
    size_m: tuple[float, float]
    sku_name: str

    @property
    def area(self) -> float:
        return self.size_m[0] * self.size_m[1]


def boxes_for(area_m2: float, sku: Sku, waste_pct: float) -> tuple[int, int]:
    tiles = math.ceil(area_m2 / sku.m2_per_tile * (1 + waste_pct / 100))
    return math.ceil(tiles / sku.tiles_per_box), tiles


def quote_by_sku(rooms: list[Room], skus: dict[str, Sku],
                 waste_pct: float = 8.0) -> dict[str, dict]:
    grouped = {}
    for r in rooms:
        if r.sku_name not in skus:
            raise KeyError(f"room {r.name!r} references unknown sku "
                           f"{r.sku_name!r}")
        g = grouped.setdefault(r.sku_name, {"area_m2": 0.0, "rooms": []})
        g["area_m2"] += r.area
        g["rooms"].append(r.name)

    out = {}
    for name, g in grouped.items():
        boxes, tiles = boxes_for(g["area_m2"], skus[name], waste_pct)
        out[name] = {"area_m2": round(g["area_m2"], 2), "rooms": g["rooms"],
                     "tiles": tiles, "boxes": boxes, "waste_pct": waste_pct}
    return out


def quote_per_room(rooms: list[Room], skus: dict[str, Sku],
                   waste_pct: float = 8.0) -> dict[str, dict]:
    """Rounds per room. Kept to demonstrate the difference, not for production."""
    out = {}
    for r in rooms:
        boxes, tiles = boxes_for(r.area, skus[r.sku_name], waste_pct)
        acc = out.setdefault(r.sku_name, {"area_m2": 0.0, "rooms": [],
                                          "tiles": 0, "boxes": 0,
                                          "waste_pct": waste_pct})
        acc["area_m2"] = round(acc["area_m2"] + r.area, 2)
        acc["rooms"].append(r.name)
        acc["tiles"] += tiles
        acc["boxes"] += boxes
    return out
```

- [x] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_quote.py -v`
Expected: 7 passed

- [x] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: 45 passed

- [x] **Step 6: Commit**

```bash
git add renderer/quote.py tests/test_quote.py
git commit -m "feat(quote): aggregate by sku before rounding boxes"
```

---

## After this plan

The bench exists so the operator can answer spec risk #1 on their own photos.
That answer decides what comes next:

- **Convincing** → next plan is the product shell: project/room CRUD on
  Postgres, client-side WebGL swap for instant tile changes, share links.
- **Not convincing** → stop and diagnose before building anything else.
  Candidates in order: glossy reflection, far-edge mipmapping, grout colour
  fidelity, and only then a segmentation model.

Sample scenes for path A (WC and kitchen, spec §4) are a separate plan and do
not block either branch.

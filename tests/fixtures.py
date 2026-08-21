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

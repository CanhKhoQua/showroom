"""Synthetic room + synthetic tile faces. THROWAWAY test input.

Round 1  : empty room     -> check the geometry is right
Round 1b : room WITH stuff -> expose the occlusion failure, then verify the fix

A synthetic scene cannot answer "would a customer believe it" -- that needs a
real photo. It can only prove the maths.
"""
import cv2
import numpy as np

W, H = 1280, 860
QUAD = [(335, 432), (952, 432), (1245, 828), (58, 828)]   # far-L, far-R, near-R, near-L

CABINET_BASE = [(800, 596), (1000, 596), (1040, 688), (820, 688)]   # footprint on floor
CABINET_HEIGHT = 178
POT_CENTRE, POT_AXES = (250, 648), (58, 20)


def offset(poly, dy):
    return [(x, y + dy) for x, y in poly]


def room(with_objects=False):
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H):                                     # wall, darkening downward
        t = y / H
        img[y, :] = (int(196 - 26 * t), int(198 - 26 * t), int(201 - 26 * t))

    rng = np.random.default_rng(3)
    floor = np.full((H, W, 3), (138, 141, 144), np.uint8)  # bare grey screed
    floor = np.clip(floor.astype(np.int16) + rng.normal(0, 7, floor.shape), 0, 255).astype(np.uint8)
    quad_mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(quad_mask, np.int32(QUAD), 255)
    img[quad_mask > 0] = floor[quad_mask > 0]

    gx = np.linspace(1.14, 0.88, W, dtype=np.float32)[None, :]   # window light, left
    gy = np.linspace(1.08, 0.94, H, dtype=np.float32)[:, None]
    img = np.clip(img.astype(np.float32) * (gx * gy)[..., None], 0, 255).astype(np.uint8)

    shade = np.zeros((H, W), np.float32)                   # soft cast shadow
    cv2.ellipse(shade, (1010, 770), (250, 100), 12, 0, 360, 1.0, -1)
    shade = cv2.GaussianBlur(shade, (0, 0), 46)
    img = np.clip(img.astype(np.float32) * (1 - 0.36 * shade[..., None]), 0, 255).astype(np.uint8)

    for a, b in ((0, 1), (0, 3), (1, 2)):                  # skirting
        cv2.line(img, QUAD[a], QUAD[b], (168, 170, 172), 3)

    if with_objects:
        img = add_objects(img)

    return np.clip(img.astype(np.float32) + rng.normal(0, 3.2, img.shape), 0, 255).astype(np.uint8)


def add_objects(img):
    contact = np.zeros((H, W), np.float32)                 # contact shadows
    cv2.ellipse(contact, (920, 694), (140, 26), 0, 0, 360, 1.0, -1)
    cv2.ellipse(contact, POT_CENTRE, (74, 22), 0, 0, 360, 1.0, -1)
    contact = cv2.GaussianBlur(contact, (0, 0), 15)
    img = np.clip(img.astype(np.float32) * (1 - 0.45 * contact[..., None]), 0, 255).astype(np.uint8)

    top = offset(CABINET_BASE, -CABINET_HEIGHT)
    front = [CABINET_BASE[3], CABINET_BASE[2], top[2], top[3]]
    side = [CABINET_BASE[2], CABINET_BASE[1], top[1], top[2]]
    cv2.fillPoly(img, [np.int32(front)], (66, 92, 122))    # wooden cabinet
    cv2.fillPoly(img, [np.int32(side)], (48, 68, 92))
    cv2.fillPoly(img, [np.int32(top)], (86, 116, 150))
    cv2.polylines(img, [np.int32(front)], True, (40, 56, 74), 2)

    # Deliberately near-grey: its colour sits close to the floor, which is what
    # makes it the hard case for any colour-based detector.
    cv2.ellipse(img, POT_CENTRE, POT_AXES, 0, 0, 360, (92, 106, 126), -1)
    cv2.rectangle(img, (POT_CENTRE[0] - 44, POT_CENTRE[1] - 74),
                  (POT_CENTRE[0] + 44, POT_CENTRE[1]), (88, 100, 120), -1)
    rng = np.random.default_rng(9)
    for _ in range(90):                                    # foliage
        angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(10, 96)
        p = (int(POT_CENTRE[0] + np.cos(angle) * radius),
             int(POT_CENTRE[1] - 96 - np.sin(angle) * radius * 0.75))
        cv2.circle(img, p, int(rng.integers(7, 17)), (58, 108, 62), -1)
    return img


def stone_face(seed, size=600):
    """One printed face of a stone-look tile: cream ground plus running veins."""
    rng = np.random.default_rng(seed)
    ground = np.full((size, size, 3), (232, 234, 236), np.uint8)
    ground = np.clip(ground.astype(np.int16) + rng.normal(0, 4, ground.shape), 0, 255).astype(np.uint8)
    veins = np.zeros((size, size), np.float32)
    for _ in range(rng.integers(5, 9)):
        x, y = rng.integers(0, size), rng.integers(0, size)
        angle = rng.uniform(0, np.pi)
        points = []
        for _ in range(rng.integers(60, 150)):
            angle += rng.normal(0, 0.22)
            x += np.cos(angle) * 7
            y += np.sin(angle) * 7
            points.append((int(x), int(y)))
        cv2.polylines(veins, [np.int32(points)], False, rng.uniform(0.30, 0.85),
                      int(rng.integers(1, 4)), cv2.LINE_AA)
    veins = cv2.GaussianBlur(veins, (0, 0), 1.5)
    return np.clip(ground.astype(np.float32) * (1 - 0.42 * veins[..., None]), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    cv2.imwrite("room_empty.png", room(with_objects=False))
    cv2.imwrite("room_objects.png", room(with_objects=True))
    for i in range(4):
        cv2.imwrite(f"face{i + 1}.png", stone_face(11 + i))
    print("quad         =", " ".join(f"{x},{y}" for x, y in QUAD))
    print("cabinet base =", " ".join(f"{x},{y}" for x, y in CABINET_BASE))
    print("pot centre   =", POT_CENTRE, "axes", POT_AXES)

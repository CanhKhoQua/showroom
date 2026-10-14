"""Metric floor plane from one photo, so tiles keep their true shape.

Vertical edges (wall corners, door jambs) meet at the vertical vanishing
point, which gives the direction of gravity in camera coordinates. With an
assumed lens and camera height that fixes the floor homography. Assumptions:
a phone main camera (26 mm equivalent) held about 1.4 m above the floor.
"""
import itertools

import cv2
import numpy as np

DEFAULT_PITCH_DEG = 45.0
MAX_EXTENT_MM = 15000.0


def camera_matrix(width: int, height: int, focal_35mm: float = 26.0) -> np.ndarray:
    focal = focal_35mm / np.hypot(36.0, 24.0) * np.hypot(width, height)
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])


def gravity_from_segments(segments: np.ndarray, K: np.ndarray,
                          max_tilt_deg: float = 30.0,
                          inlier_deg: float = 1.0) -> np.ndarray | None:
    """Downward unit vector shared by the near-vertical segments, or None."""
    K_inv = np.linalg.inv(K)
    normals, weights = [], []
    for x1, y1, x2, y2 in segments:
        if abs(x2 - x1) > abs(y2 - y1) * np.tan(np.radians(max_tilt_deg)):
            continue
        n = np.cross(K_inv @ [x1, y1, 1.0], K_inv @ [x2, y2, 1.0])
        normals.append(n / np.linalg.norm(n))
        weights.append(np.hypot(x2 - x1, y2 - y1))
    if len(normals) < 3:
        return None
    normals, weights = np.array(normals), np.array(weights)
    # A segment is consistent with g when g lies in the plane through the
    # camera centre and the segment, i.e. n . g ~ 0.
    longest = np.argsort(weights)[::-1][:40]
    limit = np.sin(np.radians(inlier_deg))
    best, best_score = None, 0.0
    for i, j in itertools.combinations(longest, 2):
        g = np.cross(normals[i], normals[j])
        if np.linalg.norm(g) < 1e-6:
            continue
        inliers = np.abs(normals @ (g / np.linalg.norm(g))) < limit
        score = weights[inliers].sum()
        if inliers.sum() >= 3 and score > best_score:
            best, best_score = inliers, score
    if best is None:
        return None
    g = np.linalg.svd(normals[best] * weights[best, None])[2][-1]
    g = g if g[1] > 0 else -g
    # The floor is only visible when the camera looks level or downward.
    return g if g[2] > -0.2 else None


def default_gravity(pitch_deg: float = DEFAULT_PITCH_DEG) -> np.ndarray:
    pitch = np.radians(pitch_deg)
    return np.array([0.0, np.cos(pitch), np.sin(pitch)])


def floor_plane(boundary: list, gravity: np.ndarray, K: np.ndarray,
                camera_height_mm: float = 1400.0, margin: float = 0.02) -> dict:
    """Perspective quad of the smallest floor rectangle around the outline.

    Quad order is FAR-LEFT, FAR-RIGHT, NEAR-RIGHT, NEAR-LEFT and floor_mm is
    the rectangle's true size, so the tile sheet is never stretched.
    """
    g = gravity / np.linalg.norm(gravity)
    forward = np.array([0, 0, 1.0]) - g[2] * g
    if np.linalg.norm(forward) < 1e-3:  # looking straight down
        forward = np.array([0, -1.0, 0]) + g[1] * g
    forward /= np.linalg.norm(forward)
    right = np.cross(g, forward)
    to_photo = K @ np.column_stack([right, forward, camera_height_mm * g])

    rays = np.linalg.inv(K) @ np.column_stack([np.float64(boundary), np.ones(len(boundary))]).T
    down = g @ rays
    hits = rays[:, down > 1e-6] * (camera_height_mm / down[down > 1e-6])
    a, b = right @ hits, forward @ hits
    # Pull points near the horizon back along their own ray to the depth
    # limit, so the sheet stays a sane size; clip_to_quad trims the rest.
    pull = np.minimum(1.0, MAX_EXTENT_MM / np.maximum(b, 1e-6))
    a, b = a * pull, b * pull
    pad_a, pad_b = margin * (a.max() - a.min()), margin * (b.max() - b.min())
    left, right_mm = a.min() - pad_a, a.max() + pad_a
    near, far = b.min() - pad_b, b.max() + pad_b

    corners = np.array([[left, far, 1], [right_mm, far, 1],
                        [right_mm, near, 1], [left, near, 1]]).T
    projected = to_photo @ corners
    quad = (projected[:2] / projected[2]).T
    return {"quad": np.round(quad, 1).tolist(),
            "floor_mm": [round(float(right_mm - left), 1), round(float(far - near), 1)],
            "floor_boundary": clip_to_quad(boundary, quad)}


def clip_to_quad(polygon: list, quad: np.ndarray) -> list:
    """Sutherland-Hodgman against the convex quad; unchanged when inside."""
    contour = np.float32(quad)
    if all(cv2.pointPolygonTest(contour, (float(x), float(y)), True) >= -1
           for x, y in polygon):
        return polygon
    orientation = np.sign(cv2.contourArea(contour, oriented=True))
    points = [np.float64(p) for p in polygon]
    for i in range(4):
        start, end = quad[i], quad[(i + 1) % 4]

        def side(p):
            return orientation * ((end[0] - start[0]) * (p[1] - start[1]) -
                                  (end[1] - start[1]) * (p[0] - start[0]))

        clipped = []
        for current, following in zip(points, points[1:] + points[:1]):
            s_cur, s_next = side(current), side(following)
            if s_cur >= 0:
                clipped.append(current)
            if (s_cur >= 0) != (s_next >= 0):
                clipped.append(current + (following - current) * s_cur / (s_cur - s_next))
        points = clipped
        if not points:
            return []
    return np.round(points, 1).tolist()


def estimate_floor_plane(base_bgr: np.ndarray, boundary: list) -> dict:
    height, width = base_bgr.shape[:2]
    K = camera_matrix(width, height)
    scale = min(1.0, 1024 / max(width, height))
    gray = cv2.cvtColor(cv2.resize(base_bgr, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    gravity = None
    if lines is not None:
        segments = lines.reshape(-1, 4) / scale
        long_enough = np.hypot(segments[:, 2] - segments[:, 0],
                               segments[:, 3] - segments[:, 1]) > 0.04 * max(width, height)
        gravity = gravity_from_segments(segments[long_enough], K)
    source = "vertical-lines" if gravity is not None else "default"
    plane = floor_plane(boundary, gravity if gravity is not None else default_gravity(), K)
    return {**plane, "source": source}

import cv2
import numpy as np

from renderer.floor_plane import (
    camera_matrix, estimate_floor_plane, floor_plane, gravity_from_segments)

SIZE = (3024, 4032)
HEIGHT_MM = 1400.0


def tilted_gravity(pitch_deg: float, roll_deg: float) -> np.ndarray:
    pitch, roll = np.radians(pitch_deg), np.radians(roll_deg)
    g = np.array([0.0, np.cos(pitch), np.sin(pitch)])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    return rz @ g


def project(K, point):
    p = K @ point
    return p[:2] / p[2]


def vertical_segments(K, g):
    """Wall corners and door jambs: 3D vertical edges standing on the floor."""
    segments = []
    for x, z in [(-900, 2500), (-300, 3200), (400, 2800), (1100, 3600), (700, 4200)]:
        foot = np.array([x, 0.0, z])
        foot += (HEIGHT_MM - foot @ g) * g
        top = foot - 1500 * g
        segments.append([*project(K, foot), *project(K, top)])
    return np.array(segments)


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1)))


def floor_point(K, g, pixel):
    ray = np.linalg.inv(K) @ [pixel[0], pixel[1], 1.0]
    return ray * HEIGHT_MM / (g @ ray)


def test_vertical_edges_recover_camera_tilt_despite_clutter():
    K = camera_matrix(*SIZE)
    g = tilted_gravity(40, 4)
    clutter = np.array([[200, 3000, 2600, 2900], [400, 3500, 2900, 3600],
                        [100, 2600, 1300, 2300], [1800, 1000, 1950, 2200]])
    segments = np.vstack([vertical_segments(K, g), clutter])

    estimate = gravity_from_segments(segments, K)

    assert estimate is not None
    assert angle_deg(estimate, g) < 0.5
    assert estimate[1] > 0


def test_too_few_vertical_edges_gives_no_estimate():
    K = camera_matrix(*SIZE)
    assert gravity_from_segments(np.array([[100, 100, 120, 900]]), K) is None


def test_plane_maps_photo_pixels_to_true_floor_distances():
    """Square tiles stay square: sheet millimetres equal real floor millimetres."""
    K = camera_matrix(*SIZE)
    g = tilted_gravity(40, 4)
    boundary = [[0, 2300], [3023, 2200], [3023, 4031], [0, 4031]]

    plane = floor_plane(boundary, g, K, camera_height_mm=HEIGHT_MM)

    width_mm, depth_mm = plane["floor_mm"]
    sheet = np.float32([[0, 0], [width_mm, 0], [width_mm, depth_mm], [0, depth_mm]])
    to_sheet = cv2.getPerspectiveTransform(np.float32(plane["quad"]), sheet)
    pixels = np.float32([[600, 3800], [2500, 3700], [1500, 2500], [2900, 2350]])
    on_sheet = cv2.perspectiveTransform(pixels[None], to_sheet)[0]
    on_floor = [floor_point(K, g, p) for p in pixels]
    for i in range(len(pixels)):
        for j in range(i + 1, len(pixels)):
            sheet_mm = np.linalg.norm(on_sheet[i] - on_sheet[j])
            floor_mm = np.linalg.norm(on_floor[i] - on_floor[j])
            assert abs(sheet_mm - floor_mm) < 0.01 * floor_mm


def test_plane_quad_contains_the_whole_floor_outline():
    K = camera_matrix(*SIZE)
    boundary = [[0, 2300], [1500, 2250], [3023, 2200], [3023, 4031], [0, 4031]]

    quad = np.float32(floor_plane(boundary, tilted_gravity(40, 4), K)["quad"])

    for point in boundary:
        assert cv2.pointPolygonTest(quad, tuple(map(float, point)), True) >= 0


def test_photo_without_straight_edges_falls_back_to_a_default_tilt():
    base = np.full((400, 300, 3), 128, np.uint8)
    boundary = [[0, 250], [299, 250], [299, 399], [0, 399]]

    plane = estimate_floor_plane(base, boundary)

    assert plane["source"] == "default"
    assert len(plane["quad"]) == 4
    assert all(v > 0 for v in plane["floor_mm"])


def test_photo_with_vertical_edges_uses_them():
    K = camera_matrix(*SIZE)
    base = np.full((SIZE[1], SIZE[0], 3), 200, np.uint8)
    for x1, y1, x2, y2 in vertical_segments(K, tilted_gravity(40, 4)):
        cv2.line(base, (int(x1), int(y1)), (int(x2), int(y2)), (30, 30, 30), 12)
    boundary = [[0, 2300], [3023, 2200], [3023, 4031], [0, 4031]]

    assert estimate_floor_plane(base, boundary)["source"] == "vertical-lines"


def test_outline_reaching_the_horizon_is_clipped_to_the_plane():
    """A far wall 30+ m away cannot fit a finite sheet; the plane trims it."""
    K = camera_matrix(*SIZE)
    horizon = K[1, 2]
    boundary = [[0, horizon + 3], [3023, horizon + 3], [3023, 4031], [0, 4031]]

    plane = floor_plane(boundary, tilted_gravity(0, 0), K)

    quad = np.float32(plane["quad"])
    assert max(plane["floor_mm"]) <= 15000 * 1.05
    assert len(plane["floor_boundary"]) >= 3
    for point in plane["floor_boundary"]:
        assert cv2.pointPolygonTest(quad, tuple(map(float, point)), True) >= -1
    assert min(p[1] for p in plane["floor_boundary"]) > horizon + 3


def test_outline_inside_the_plane_is_returned_unchanged():
    K = camera_matrix(*SIZE)
    boundary = [[0, 2300], [1500, 2250], [3023, 2200], [3023, 4031], [0, 4031]]

    plane = floor_plane(boundary, tilted_gravity(40, 4), K)

    assert plane["floor_boundary"] == boundary

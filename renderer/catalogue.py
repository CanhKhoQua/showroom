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

"""Aggregate room quantities by SKU before rounding to whole boxes."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING

from renderer.catalogue import Sku


@dataclass
class Room:
    name: str
    kind: str
    size_m: tuple[float, float]
    sku_name: str

    @property
    def area(self) -> float:
        return self.size_m[0] * self.size_m[1]


def boxes_for(area_m2: float, sku: Sku, waste_pct: float) -> tuple[int, int]:
    area, waste = Decimal(str(area_m2)), Decimal(str(waste_pct))
    width, depth = (Decimal(str(value)) for value in sku.tile_mm)
    if any(not value.is_finite() or value < 0 for value in (area, waste)):
        raise ValueError("area and waste must be finite and non-negative")
    if any(not value.is_finite() or value <= 0 for value in (width, depth)):
        raise ValueError("tile dimensions must be finite and positive")
    if sku.tiles_per_box <= 0:
        raise ValueError("tiles per box must be positive")
    tile_area = width * depth / 1_000_000
    tiles = int((area * (1 + waste / 100) / tile_area).to_integral_value(
        rounding=ROUND_CEILING))
    return (tiles + sku.tiles_per_box - 1) // sku.tiles_per_box, tiles


def quote_by_sku(rooms: list[Room], skus: dict[str, Sku],
                 waste_pct: float = 8.0) -> dict[str, dict]:
    grouped = {}
    for room in rooms:
        if room.sku_name not in skus:
            raise KeyError(f"room {room.name!r} references unknown sku "
                           f"{room.sku_name!r}")
        width, depth = (Decimal(str(value)) for value in room.size_m)
        if any(not value.is_finite() or value <= 0 for value in (width, depth)):
            raise ValueError("room dimensions must be finite and positive")
        group = grouped.setdefault(room.sku_name,
                                   {"area_m2": Decimal(0), "rooms": []})
        group["area_m2"] += width * depth
        group["rooms"].append(room.name)

    result = {}
    for name, group in grouped.items():
        boxes, tiles = boxes_for(group["area_m2"], skus[name], waste_pct)
        result[name] = {"area_m2": float(group["area_m2"]),
                        "rooms": group["rooms"], "tiles": tiles,
                        "boxes": boxes, "waste_pct": waste_pct}
    return result


def quote_per_room(rooms: list[Room], skus: dict[str, Sku],
                   waste_pct: float = 8.0) -> dict[str, dict]:
    """Compare per-room rounding with the grouped purchase quantities."""
    result = {}
    for room in rooms:
        line = quote_by_sku([room], skus, waste_pct)[room.sku_name]
        total = result.setdefault(room.sku_name,
                                  {"area_m2": 0, "rooms": [], "tiles": 0,
                                   "boxes": 0, "waste_pct": waste_pct})
        total["area_m2"] += line["area_m2"]
        total["rooms"].extend(line["rooms"])
        total["tiles"] += line["tiles"]
        total["boxes"] += line["boxes"]
    return result

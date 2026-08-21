"""Multi-room quote: does aggregating by SKU before rounding actually save money?

THROWAWAY probe. Customers here buy for the whole house at once, so the same
SKU lands in several rooms. Rounding boxes per-room wastes a partial box each
time; rounding once per SKU wastes it once.
"""
import math
from dataclasses import dataclass


@dataclass
class Sku:
    name: str
    tile_mm: tuple          # (width, depth)
    tiles_per_box: int

    @property
    def m2_per_tile(self):
        return (self.tile_mm[0] / 1000) * (self.tile_mm[1] / 1000)


@dataclass
class Room:
    name: str
    kind: str               # living | bedroom | wc | kitchen
    size_m: tuple           # (width, depth) in metres
    sku: Sku

    @property
    def area(self):
        return self.size_m[0] * self.size_m[1]


def boxes_for(area_m2, sku, waste_pct):
    tiles = math.ceil(area_m2 / sku.m2_per_tile * (1 + waste_pct / 100))
    return math.ceil(tiles / sku.tiles_per_box), tiles


def per_room_rounding(rooms, waste_pct):
    total = {}
    for r in rooms:
        b, _ = boxes_for(r.area, r.sku, waste_pct)
        total[r.sku.name] = total.get(r.sku.name, 0) + b
    return total


def per_sku_rounding(rooms, waste_pct):
    area = {}
    for r in rooms:
        area[r.sku.name] = area.get(r.sku.name, 0) + r.area
    return {name: boxes_for(a, next(r.sku for r in rooms if r.sku.name == name), waste_pct)[0]
            for name, a in area.items()}


if __name__ == "__main__":
    # Realistic Vietnamese townhouse, one purchase for the whole place.
    granite  = Sku("granite-800",  (800, 800), 3)
    wood600  = Sku("wood-look-600", (600, 600), 4)
    kitchen6 = Sku("kitchen-600",  (600, 600), 4)
    wc300    = Sku("wc-300",       (300, 300), 11)

    rooms = [
        Room("phong khach", "living",  (4.2, 3.8), granite),
        Room("ngu 1",       "bedroom", (3.6, 3.2), wood600),
        Room("ngu 2",       "bedroom", (3.2, 3.0), wood600),
        Room("ngu 3",       "bedroom", (3.0, 3.0), wood600),
        Room("bep",         "kitchen", (3.0, 3.5), kitchen6),
        Room("wc 1",        "wc",      (1.8, 2.2), wc300),
        Room("wc 2",        "wc",      (1.6, 2.0), wc300),
    ]
    WASTE = 8.0

    a = per_room_rounding(rooms, WASTE)
    b = per_sku_rounding(rooms, WASTE)

    print(f"{'SKU':>15} | {'rooms':>5} | {'area m2':>8} | {'per-room':>8} | {'per-sku':>7} | saved")
    print("-" * 68)
    saved_total = 0
    for name in a:
        rs = [r for r in rooms if r.sku.name == name]
        area = sum(r.area for r in rs)
        saved = a[name] - b[name]
        saved_total += saved
        print(f"{name:>15} | {len(rs):5d} | {area:8.2f} | {a[name]:8d} | {b[name]:7d} | {saved:+d}")
    print("-" * 68)
    print(f"{'TOTAL boxes':>15} | {'':5} | {sum(r.area for r in rooms):8.2f} | "
          f"{sum(a.values()):8d} | {sum(b.values()):7d} | {saved_total:+d}")

    # Also: how many rooms use each render path?
    photo = [r for r in rooms if r.kind in ("living", "bedroom")]
    sample = [r for r in rooms if r.kind in ("wc", "kitchen")]
    print(f"\nrender path split: real-photo {len(photo)} rooms "
          f"({sum(r.area for r in photo):.1f} m2) | "
          f"sample-scene {len(sample)} rooms ({sum(r.area for r in sample):.1f} m2)")

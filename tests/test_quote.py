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
    assert Room("r", "bedroom", (3.0, 4.0), WOOD.name).area == pytest.approx(12)


def test_boxes_for_includes_waste_and_rounds_up():
    assert boxes_for(11.52, WOOD, 8) == (9, 35)


def test_zero_waste_is_honoured():
    assert boxes_for(3.6, WOOD, 0) == (3, 10)


def test_aggregating_by_sku_never_costs_more_than_per_room():
    grouped = quote_by_sku(TOWNHOUSE, SKUS)
    separate = quote_per_room(TOWNHOUSE, SKUS)
    assert all(grouped[name]["boxes"] <= separate[name]["boxes"]
               for name in grouped)


def test_measured_townhouse_totals():
    assert sum(v["boxes"] for v in quote_per_room(TOWNHOUSE, SKUS).values()) == 50
    assert sum(v["boxes"] for v in quote_by_sku(TOWNHOUSE, SKUS).values()) == 48


def test_by_sku_reports_shared_rooms_and_total_area():
    line = quote_by_sku(TOWNHOUSE, SKUS)[WOOD.name]
    assert line["rooms"] == ["bed 1", "bed 2", "bed 3"]
    assert line["area_m2"] == pytest.approx(30.12)


def test_unknown_sku_is_an_error():
    with pytest.raises(KeyError, match="unknown sku"):
        quote_by_sku([Room("r", "bedroom", (3, 3), "missing")], SKUS)


def test_exact_box_boundary_does_not_add_a_box_from_float_roundoff():
    tile = Sku("rectangular", (300, 700), [], 3, "matte")
    rooms = [Room("r", "living", (0.7, 0.9), tile.name)]
    assert quote_by_sku(rooms, {tile.name: tile}, 0)[tile.name]["boxes"] == 1
    assert boxes_for(0.63, tile, 0) == (1, 3)


def test_area_is_not_rounded_before_calculating_boxes():
    tile = Sku("square", (100, 100), [], 1, "matte")
    rooms = [Room("a", "living", (0.101, 0.1), tile.name),
             Room("b", "living", (0.101, 0.1), tile.name)]
    line = quote_by_sku(rooms, {tile.name: tile}, 0)[tile.name]
    assert line["boxes"] == 3


def test_empty_project_has_no_quote_lines():
    assert quote_by_sku([], SKUS) == {}


@pytest.mark.parametrize("area,waste", [(-1, 8), (1, -1), (float("nan"), 8),
                                        (float("inf"), 8), (1, float("nan"))])
def test_invalid_area_or_waste_is_rejected(area, waste):
    with pytest.raises(ValueError):
        boxes_for(area, WOOD, waste)


@pytest.mark.parametrize("tile_mm,tiles_per_box", [((0, 600), 4),
                                                   ((600, 600), 0)])
def test_invalid_packaging_is_rejected(tile_mm, tiles_per_box):
    with pytest.raises(ValueError):
        boxes_for(12, Sku("bad", tile_mm, [], tiles_per_box, "matte"), 8)

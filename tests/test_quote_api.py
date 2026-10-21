import pytest
from fastapi.testclient import TestClient

from bench.app import app


client = TestClient(app)


def project():
    return {
        "rooms": [
            {"name": "Bedroom 1", "size_m": [3, 3], "sku_name": "WOOD"},
            {"name": "Bedroom 2", "size_m": [3, 3], "sku_name": "WOOD"},
        ],
        "skus": {
            "WOOD": {"tile_mm": [600, 600], "tiles_per_box": 4,
                     "price_per_box": 240000, "waste_pct": 8},
        },
        "dimensions_confirmed": True,
    }


def test_quote_groups_rooms_before_rounding_and_prices_whole_boxes():
    response = client.post("/quote", json=project())
    assert response.status_code == 200
    result = response.json()
    line = result["lines"][0]
    assert line["sku_name"] == "WOOD"
    assert line["rooms"] == ["Bedroom 1", "Bedroom 2"]
    assert line["area_m2"] == 18
    assert line["tiles"] == 54
    assert line["boxes"] == 14
    assert line["purchased_m2"] == 20.16
    assert line["amount"] == 3360000
    assert result["total"] == 3360000
    assert result["currency"] == "VND"


def test_quote_uses_each_skus_packaging_price_and_waste():
    body = project()
    body["skus"]["WOOD"]["waste_pct"] = 0
    body["skus"]["STONE"] = {"tile_mm": [1000, 1000], "tiles_per_box": 2,
                              "price_per_box": 500000, "waste_pct": 10}
    body["rooms"].append({"name": "Kitchen", "size_m": [2, 2],
                          "sku_name": "STONE"})
    result = client.post("/quote", json=body).json()
    assert [line["boxes"] for line in result["lines"]] == [13, 3]
    assert result["total"] == 4620000


def test_quote_requires_measured_dimensions_to_be_confirmed():
    body = project()
    body["dimensions_confirmed"] = False
    response = client.post("/quote", json=body)
    assert response.status_code == 422


def test_quote_rejects_unknown_sku_with_a_useful_message():
    body = project()
    body["rooms"][0]["sku_name"] = "MISSING"
    response = client.post("/quote", json=body)
    assert response.status_code == 422
    assert "MISSING" in response.json()["detail"]


@pytest.mark.parametrize("field,value", [
    ("tiles_per_box", 0), ("tiles_per_box", 3.5),
    ("price_per_box", -1), ("price_per_box", 1.5),
    ("waste_pct", -1), ("waste_pct", "NaN"),
    ("tile_mm", [0, 600]), ("tile_mm", [600]),
])
def test_quote_rejects_invalid_sku_data(field, value):
    body = project()
    body["skus"]["WOOD"][field] = value
    assert client.post("/quote", json=body).status_code == 422


@pytest.mark.parametrize("size", [[0, 3], [-2, 3], [3], ["Infinity", 3]])
def test_quote_rejects_invalid_room_dimensions(size):
    body = project()
    body["rooms"][0]["size_m"] = size
    assert client.post("/quote", json=body).status_code == 422


def test_quote_requires_at_least_one_room():
    body = project()
    body["rooms"] = []
    assert client.post("/quote", json=body).status_code == 422


def test_zero_price_and_zero_waste_are_valid_explicit_values():
    body = project()
    body["skus"]["WOOD"].update(price_per_box=0, waste_pct=0)
    result = client.post("/quote", json=body).json()
    assert result["lines"][0]["tiles"] == 50
    assert result["total"] == 0

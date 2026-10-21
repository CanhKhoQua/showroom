"""Local estimates from measured room sizes and operator-supplied prices."""
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from renderer.catalogue import Sku
from renderer.quote import Room, quote_by_sku


router = APIRouter()
Dimension = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QuoteRoom(BaseModel):
    name: Name
    size_m: tuple[Dimension, Dimension]
    sku_name: Name


class QuoteSku(BaseModel):
    tile_mm: tuple[Dimension, Dimension]
    tiles_per_box: Annotated[int, Field(gt=0, strict=True)]
    price_per_box: Annotated[int, Field(ge=0, strict=True)]
    waste_pct: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class QuoteRequest(BaseModel):
    rooms: Annotated[list[QuoteRoom], Field(min_length=1)]
    skus: dict[Name, QuoteSku]
    dimensions_confirmed: Literal[True]


@router.post("/quote")
def create_quote(body: QuoteRequest):
    rooms = [Room(r.name, "", r.size_m, r.sku_name) for r in body.rooms]
    unknown = {r.sku_name for r in rooms} - body.skus.keys()
    if unknown:
        raise HTTPException(422, f"Unknown SKU: {', '.join(sorted(unknown))}")

    lines = []
    for name in dict.fromkeys(r.sku_name for r in rooms):
        spec = body.skus[name]
        sku = Sku(name, spec.tile_mm, [], spec.tiles_per_box, "")
        matching = [r for r in rooms if r.sku_name == name]
        line = quote_by_sku(matching, {name: sku}, spec.waste_pct)[name]
        width, depth = (Decimal(str(value)) for value in spec.tile_mm)
        purchased = width * depth / 1_000_000 * spec.tiles_per_box * line["boxes"]
        lines.append({**line, "sku_name": name,
                      "tile_mm": spec.tile_mm,
                      "tiles_per_box": spec.tiles_per_box,
                      "purchased_m2": float(purchased),
                      "price_per_box": spec.price_per_box,
                      "amount": line["boxes"] * spec.price_per_box})
    return {"lines": lines, "total": sum(line["amount"] for line in lines),
            "currency": "VND"}

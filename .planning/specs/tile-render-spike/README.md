# Spike: project tile texture onto a floor photo — THROWAWAY

Not product code. It exists to answer one question:
**is a geometry-only render convincing enough to sell from?**

```
.venv/bin/python synth_scene.py
.venv/bin/python tile_render.py --photo room_objects.png \
  --faces face1.png,face2.png,face3.png,face4.png \
  --quad "FAR-L FAR-R NEAR-R NEAR-L" \
  --floor-mm 4200x3200 --auto-mask colour-deviation --threshold 13 --out out.png
```

## Round 1 — geometry: CONFIRMED

No AI involved. All of this is plain OpenCV.

- homography from 4 clicked points — grout lines converge correctly, rows
  compress with depth
- true physical scale — 7 × 600 mm tiles across a 4.2 m floor
- 4-face variation with a fixed seed — no visible repeat, reopens identically
- grout width and colour as parameters
- relighting — the room's cast shadow and window gradient transfer onto the
  new floor
- the same measurements produce the quote: 13.44 m² → 41 tiles → 11 boxes

Bug found and fixed: **blown highlights.** A bright texture (232–236) times a
gain of 1.75 exceeded 255 and hard-clipped, erasing the veining. Fixed by
narrowing gain to [0.55, 1.35] plus a soft highlight roll-off (`soft_clip`,
knee 0.80).

## Round 1b — occlusion: the real constraint

Anything standing on the floor gets painted over unless it is masked out.

Bug found and fixed: **morphology was applied to the floor mask.** A handful of
floor-looking pixels inside the cabinet dilated outward, merged, and swallowed
the whole object. Measured: 11.6% → 91.1% of the cabinet misclassified as floor
by a single `MORPH_CLOSE`. Morphology must act on the OBJECTS, never the floor.

GrabCut was tried and **fails by construction**: its background seed is
everything outside the quad, which is wall — the same grey as the floor. Two
identical colour models discriminate nothing. Kept in the code so the failure
stays reproducible.

Replaced with a one-class model: learn floor colour from a strip along the near
edge, flag deviating pixels. Lightness weighted at 0.15 so shadows survive
(measured: lit floor d²=0.3, shadowed floor d²=0.5).

### Threshold sweep — and why there is no good answer

| threshold | cabinet kept | pot kept | bare floor tiled |
|---|---|---|---|
| none | 0.0% | 0.8% | 100% |
| **13** | **100%** | 23.4% | **100%** |
| 6 | 100% | 99.8% | 37% |
| 3 | 100% | 100% | 0% |

Only 13 is usable, and it misses the pot. The pot was deliberately made
near-grey: at d²=5.4 it sits between bare floor (0.5) and the wooden cabinet
(19–23). Lower the threshold to catch it and real floor starts being rejected —
threshold 6 shreds 63% of the floor.

**Conclusion: a global colour threshold cannot separate a near-neutral object
from a neutral floor. No threshold exists that does both.**

## What this means for the design

1. Geometry is solved and needs no GPU. Ship it.
2. Automatic masking works only for strongly coloured objects. It cannot work
   for near-neutral ones — and **white sanitaryware on tile is exactly that
   case**, i.e. the bathroom, i.e. half the catalogue. Colour-based masking is
   dead on arrival for bathrooms.
3. So v1 must ship the **hand-drawn** exclusion path as the primary mechanism,
   with auto as a convenience for easy cases only.
4. Making auto reliable requires a real segmentation model (SAM 2 or an
   ADE20K-trained segmenter), which is where GPU cost enters. We now know
   precisely what that spend buys: object cut-outs, not tile rendering.
5. Hand-drawing must be **free-form**, not rectangles. Boxes over-exclude and
   leave visible grey patches of untiled floor around each object.

## Still unanswered — needs a real photo

Realism. A synthetic grey plane in a void cannot tell us whether a customer
believes the result. Also untested: glossy tile reflections, and aliasing along
the far edge where tiles compress hardest (needs mipmapping).

Needs: 1–2 phone photos of finished rooms, and 1–2 flat-on texture shots with
the tile size in mm.

## Round 1c — multi-room quote, and how occlusion gets scoped away

Customers here buy for the whole house in one go, so a project is N rooms, not
one room. Two consequences, both measured on a realistic townhouse (7 rooms,
63.74 m², 4 SKUs) — see `quote.py`.

### Rounding boxes per SKU instead of per room

| | boxes |
|---|---|
| round per room | 50 |
| round per SKU across rooms | 48 |

**Saving is 2 boxes / 4% — real but too small to sell on.** Do not build a
"saves the customer money" story on this.

The actual value is batch consistency: ordering 23 boxes as ONE line makes
same-lot fulfilment natural. Three separate per-room lines can be filled from
three production lots, so three bedrooms on "the same" tile come out in three
shades. That is the expensive failure, not the 2 boxes.

### Occlusion is bounded by scope, not solved by AI

Splitting the render path by room kind, on the same 7-room house:

```
real photo (path B)   : 4 rooms — 46.1 m²  -> 72% of area
sample scene (path A) : 3 rooms — 17.7 m²  -> 28% of area
```

Bathrooms and kitchens are simultaneously where colour masking fails (white
fixtures on light tile) AND where a photo of the customer's own room is worth
least (bathrooms look alike). Route those to pre-built sample scenes and the
failure mode disappears instead of needing a segmentation model.

**Conclusion: no torch, no model download, no GPU for v1.** Segmentation moves
to nice-to-have for a later phase.

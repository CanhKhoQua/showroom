# Showroom render bench

A local tool for testing floor and wall tile renders on room photos.

## Run with Docker

Docker with the Compose plugin is required. Python dependencies are installed
inside the image.

```bash
docker compose up --build -d --wait
```

Open <http://localhost:8100> and click **Try demo** to load a room preset and
render it with a sample tile. Choose another tile from the catalogue to compare
looks. Catalogue samples are illustrative textures, not product photographs.

The **Choose a room** gallery includes dining, living, bedroom, kitchen and
bathroom presets. Selecting a room creates a preview with the current material.
Each preset includes an editable floor outline or perspective plane and object
protection, so no live model inference is needed to try it. Switching rooms
resets the previous outline and preview; custom texture selections are retained.
Floor dimensions in the presets are illustrative estimates, not measurements
for a quote. Uploading a customer photo clears the preset selection.

To use your own photo, click **Upload room**, choose **Point select**, and click
**Include area** before clicking inside the floor or wall. Switch to
**Exclude object** and click rugs, furniture, or adjoining surfaces that should
remain unchanged. Add points to refine the selection; Undo removes the last
point. EdgeSAM runs on the local CPU and reuses the current photo embedding.
Review the outline before creating a preview.

Choose **Add floor** or **Add wall** to work on another surface in the same
photo. Each surface retains its own selection, object protection, material,
custom texture faces, dimensions, tile direction, and perspective. Switch
between them using **Editing surface**. Create preview combines all started
surfaces; untouched empty surfaces are skipped. Remove surface deletes only
the selected layer. Projects store all surfaces and reopen the active one.

Use **Perspective** to mark four corners in this
order: far-left, far-right, near-right, near-left. Drag the handles to align
the guide grid with the room's floor edges. Corners may extend outside the
photo when the floor plane continues beyond its crop. For an L-shaped floor
or other uneven edges, choose **Outline**, click at each bend, then
choose **Finish outline**, press Enter, or click the first point to close it.
The outline can have more than four points. Drag a point or use the arrow
keys to refine it; Undo reopens the outline for adding points. If you trace
an outline first, the bench starts with a rectangular perspective plane;
this plane follows outline edits until you adjust it using **Perspective**.
Keep the perspective corners outside the entire outline so all selected floor
is covered. Use **Protect objects**
to draw a closed freehand area over an
object that should retain its original appearance. You can also upload custom
texture faces and set the tile dimensions in millimetres and floor dimensions
in metres.
For rectangular tiles, **Half offset** staggers joints along the longer side,
including upright wood planks, while retaining the source grain direction.
**Tile direction** turns the material by 90 degrees while retaining physical
tile dimensions. Enter the measured width and depth of a floor, or width and
height of a wall. The initial wall perspective is a selection bounding box;
align its corners with the actual wall plane before judging the tile shape.

Texture interpolation, illumination, and edge blending use linear light.
Low-frequency lighting and contact shadows come from the original surface;
unselected pixels remain unchanged. Float processing is restricted to the
selected surface bounds to reduce memory and CPU use on large photos.

The two [EdgeSAM ONNX weights](https://github.com/chongzhou96/EdgeSAM) total
about 36 MiB and are pinned by revision and SHA-256 in the Dockerfile. They run
with ONNX Runtime's CPU provider using two threads, without PyTorch, CUDA, or an
external image service. The [S-Lab license](https://github.com/chongzhou96/EdgeSAM/blob/master/LICENSE)
permits non-commercial use; this bench uses the models for evaluation. The
complete license is in `renderer/edgesam-license.txt`.

For an automatic starting point, upload the room and click **Detect floor**.
SegFormer B4 runs locally on two CPU threads and returns an editable outline of
the dominant floor region, with detected object holes protected. Review the
outline, especially rugs and object edges, before rendering. The outline edge
is snapped to colour edges in the photo, because the model itself only predicts
at a quarter of its input resolution. Confident non-floor predictions remain
protected during refinement.

Preview texture resolution follows the photo geometry and available source
detail, with a roughly 16-megapixel tile sheet limit, to retain grain detail.

When no Perspective corners are set, detection also estimates the floor plane:
vertical edges such as wall corners and door frames give the camera tilt, and
an assumed phone lens (26 mm equivalent) held 1.4 m above the floor gives the
scale. The plane fills **Surface dimensions** with its estimated size, so tiles keep
their shape; changing those values afterwards stretches the tiles. Without
usable vertical edges the tilt is a 45° guess. Check the result in
**Perspective**. Repeating detection retains hand-drawn object protection and
any Perspective corners already set.

The model is downloaded during the Docker build (approximately 258 MB); no
photo is sent to a model service, and inference needs no internet or API key.
On the tested room photo, warm inference took about 1.4 seconds and peak process
memory was about 1.8 GiB. B4 was selected because B0 classified the gray rugs as
floor; outlines still need review on other photos.
The pinned [ONNX export](https://huggingface.co/Xenova/segformer-b4-finetuned-ade-512-512)
is derived from [NVIDIA SegFormer B4 on ADE20K](https://huggingface.co/nvidia/segformer-b4-finetuned-ade-512-512).
Its [NVIDIA license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)
limits use to non-commercial research or evaluation; this experimental bench
uses it for evaluation. The complete license is in `renderer/segformer-license.txt`.

HEIC and HEIF room photos and texture images are decoded inside Docker and
converted to an upright PNG automatically before editing. JPG, PNG and WebP
images can also be uploaded directly.

Click **Create preview**, use the before/after slider to compare the result with the
original photo, then download the rendered image.

The bench is bound to localhost by default. Rebuild the image after changing
source code. See **Shared server projects** below for access from the local network.

Demo room photo: [Unsplash](https://images.unsplash.com/photo-1600607687920-4e2a09cf159d?auto=format&fit=crop&w=1400&q=85),
used under the [Unsplash License](https://unsplash.com/license).

The four additional room illustrations in `bench/static/rooms/` were generated
with the built-in image generation tool. The prompts describe neutral modern
living, bedroom, kitchen and bathroom interiors with broad unobstructed matte
floors, natural daylight and no people, rugs, branding or text. Full images are
1400 px wide; separate 280 px thumbnails keep the gallery lightweight. The
stored perspective assumes a 26 mm equivalent lens, 1.4 m camera height and a
20-degree downward tilt. The bathroom preset protects the glass shower tray.

## Sample material sources

The four illustrative texture samples are stored in
`bench/static/materials/catalogue.json` with publisher links, original download
locations and CC0-1.0 licence metadata. This file-based catalogue requires no
database. The sample texture files may be redistributed, modified and used
commercially under CC0; the AI model has its own separate licence above.
There are no manufacturer product codes, prices or showroom stock claims.
Selected materials link to their original asset pages. Tile dimensions are
preview presets; they do not certify an actual product specification.

Each illustrative material has four colour-map crops in `bench/static/materials/`,
numbered `01` to `04`. These are illustrative material samples, not manufacturer
product photographs. The original colours and grain direction are retained;
only cropping, JPEG encoding, and proportional oak downsampling are applied.
The limestone and stone atlas crops contain a single face with all source grout
removed, so the renderer can draw its own joints.

| Files | Source and original download | Faces |
| --- | --- | --- |
| `limestone-*.jpg` | [ambientCG Tiles 139](https://ambientcg.com/view?id=Tiles139), procedural with bitmap elements; [2K JPG pack](https://ambientcg.com/get?file=Tiles139_2K-JPG.zip), `Tiles139_2K-JPG_Color.jpg` | Four 328 × 328 px faces |
| `concrete-*.jpg` | [ambientCG Concrete 048](https://ambientcg.com/view?id=Concrete048), surface photogrammetry; [1K JPG pack](https://ambientcg.com/get?file=Concrete048_1K-JPG.zip), `Concrete048_1K-JPG_Color.jpg` | Four 512 × 512 px quadrants |
| `stone-*.jpg` | [ambientCG Tiles 138](https://ambientcg.com/view?id=Tiles138), surface photogrammetry; [4K JPG pack](https://ambientcg.com/get?file=Tiles138_4K-JPG.zip), `Tiles138_4K-JPG_Color.jpg` | Four 280 × 280 px dark stone faces; stone species is unspecified |
| `oak-*.jpg` | [Poly Haven Oak Veneer 01](https://polyhaven.com/a/oak_veneer_01) by Jenelle van Heerden; [2K diffuse map](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/oak_veneer_01/oak_veneer_01_diff_2k.jpg) | Four 213 × 1344 px vertical crops, downsampled to 152 × 960 px |

All these assets use [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/):
see the official [ambientCG license](https://docs.ambientcg.com/license/) and
[Poly Haven license](https://polyhaven.com/license).

The limestone source covers approximately 2 × 2 m with six tiles along each
axis, or about 333 mm per original tile. The oak source covers 1.83 × 1.83 m;
each crop represents approximately 190 × 1200 mm at its native grain scale.
The concrete and dark stone publishers do not specify physical dimensions.
Catalogue tile dimensions are rendering presets and do not certify a product.

## Multi-room floor estimates

The **Floor estimate** panel combines measured floors by product code before
rounding to whole boxes. Set the floor dimensions in metres and actual tile
dimensions in millimetres in the material panel, then enter a room name, product
code, tiles per box, price per box in VND and waste percentage. Confirm that the
dimensions are measured, then choose **Add floor to estimate** for each room.
The 8% waste starting value is editable; confirm it for the product and laying
pattern. For irregular floors, add measured rectangular sections separately.

Room presets and automatic detection provide estimated dimensions, which must
be replaced or verified against measurements before quoting. Changing the room,
material or dimensions clears the measurement confirmation. Floors already
added keep a snapshot of their photo, outline, protected objects, material and
dimensions. Select an added floor to reopen it. Remove and re-add a floor to
revise its quote quantities. Rooms with the same product code must use the same tile size, packaging,
price and waste; enter a distinct code for a different specification.

The estimate lists required tiles, whole boxes, purchased area and the total
material cost. Prices apply to whole boxes. Delivery, installation and any
additional taxes or fees are excluded. The sample materials have no product
prices or packaging defaults; enter actual supplier data. Use **Save project**
to keep the rooms and estimate on the server, or **Print estimate** to print or
save a PDF.

`POST /quote` accepts JSON with `rooms` (`name`, `size_m`, `sku_name`), `skus`
(keyed by product code, with `tile_mm`, `tiles_per_box`, `price_per_box`,
`waste_pct`) and `dimensions_confirmed: true`. It returns grouped `lines`,
`total` and `currency: "VND"`. Missing product codes, unconfirmed dimensions,
non-finite or non-positive sizes, invalid box counts and negative prices or
waste are rejected with HTTP 422. Tile quantities use decimal arithmetic so an
exact box boundary does not acquire an extra tile through floating-point error.

## Shared server projects

Enter a project name and choose **Save project**. The server stores the current
room, uploaded photo and textures, editable floor geometry, and all added floors
with their quote data. The project link restores this information after a reload
or from a separate browser or device. A saved preview can be recreated with
**Create preview**. Edits require another save; the page shows unsaved changes
and asks before leaving them. **New project** opens a fresh workspace, and
**Save a copy** keeps a separate version under a new link.

Projects are stored in SQLite inside the `showroom-data` Docker volume. Rebuilding
or recreating the container and `docker compose down` preserve that volume.
For a direct Python launch, `SHOWROOM_DATA_DIR` selects the data directory and
defaults to `data/`.

To use phones and tablets on the same Wi-Fi or LAN, put these settings in a local
`.env` file, replacing the example address with the server computer's LAN address:

```dotenv
SHOWROOM_BIND_ADDRESS=0.0.0.0
SHOWROOM_PUBLIC_ORIGIN=http://192.168.1.10:8100
```

Then run `docker compose up -d --wait` and open that LAN address on each device.
The computer must remain running and reachable on the same network. The public
origin makes **Copy link** use the LAN address even when the server computer
opens the app through localhost. Update it if the computer's address changes.
When clipboard access is unavailable over LAN HTTP, the button selects the link
for manual copying.

Anyone with a project link can view and edit that project. Project links are
random and projects are not publicly listed. If two devices edit the same saved
revision, the first save succeeds and the second is rejected without overwriting
it. The second device can reload the saved version or choose **Save a copy** to
keep its edits. This setup serves the local network; Internet hosting and user
accounts are separate work.

`POST /projects` creates a named snapshot and returns its link and revision.
`GET /projects/{id}` opens it, and `PUT /projects/{id}` requires the last loaded
`revision`. A stale revision returns HTTP 409. Project responses use `no-store`
cache control, and the server validates room dimensions, product references,
finite floor coordinates and embedded raster image data before saving.

## Test

```bash
docker compose run --rm --no-deps bench python -m pytest -q
```

The browser checks require a Playwright package with Chromium and WebKit
installed on the host. Run `node tests/browser_projects.cjs`, followed by
`node tests/browser_project_loading.cjs`, against the running local server.
If Playwright is installed elsewhere, set `PLAYWRIGHT_MODULE` to its package
directory. `SHOWROOM_TEST_URL` can select another local server address.
These checks create sample server projects and save screenshots and their
test fixture in the ignored `artifacts/server-project-check/` directory.
They cover save/reload, separate browser sessions, LAN share links, reopening
rooms, conflicting saves, save-copy recovery, WebKit mobile rendering and
blocking edits while project data or the quote is still loading.

`node tests/browser_surfaces.cjs` covers CPU click selection, exclusion and undo,
independent floor/wall settings, combined previews, save-copy/reload, PNG export,
and WebKit touch interaction. Set `SHOWROOM_TEST_PROJECT` to an existing photo
project ID to use it as a fixture; the check saves a copy and keeps the source
unchanged. `node tests/browser_surface_races.cjs` exercises delayed script
initialization and custom-texture ownership during decoding.

## Logs and shutdown

```bash
docker compose logs -f bench
docker compose down
```

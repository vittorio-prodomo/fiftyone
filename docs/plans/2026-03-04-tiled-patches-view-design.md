# Tiled Patches View Design

## Problem

FiftyOne's patches view creates one virtual crop per annotation, showing that
annotation alone. There is no way to tile an image into a grid of fixed-size
regions and view all annotations within each tile. Additionally, the 0.2 zoom
padding (`zoomPad`) applied to patches is hardcoded with no UI control.

## Solution

Add a `to_tiles()` API that generates synthetic tile bounding boxes over
images, delegates to the existing patches view machinery with
`other_fields=True`, and returns a standard `PatchesView` where each sample is
one tile with all real annotations visible. The frontend's existing coordinate
transform (`t()` in `overlays/util.ts`) handles visual clipping naturally —
labels outside the visible tile region draw off-canvas.

## Architecture

Two layers of changes:

### Python Backend

**New `to_tiles()` method** on `SampleCollection`:

```python
view = dataset.to_tiles(
    tile_size=(640, 640),  # (width_px, height_px)
    overlap=0,  # pixels (>= 1) or fraction of tile size (< 1)
    other_fields=True,  # bring all real label fields (default True)
    min_coverage=0.0,  # skip edge tiles below this coverage fraction
)
```

**Internal flow:**

1. Read `metadata.width`/`metadata.height` from each sample (error if missing)
2. Compute grid of `Detection` bounding boxes in normalized [0,1] coordinates
3. Store as `_tile_regions` Detections field on each sample
4. Call `to_patches("_tile_regions", other_fields=other_fields)` internally
5. Return the resulting `PatchesView`

**Implementation:**

- `ToTiles` ViewStage in `fiftyone/core/stages.py` (wraps `ToPatches`)
- `to_tiles()` convenience method in `fiftyone/core/collections.py`
- Tile grid generation helper in `fiftyone/core/patches.py` or new utility

**Overlap auto-detection:** values >= 1 treated as pixels, values < 1 as
fraction of tile size.

**Metadata requirement:** Samples must have `metadata` populated. Raise clear
error if missing.

### Frontend

**zoomPad control:** Add a slider in the grid action bar that binds to
`state.options.zoomPad`. The property already exists in `ImageOptions`
(state.ts:193) and is used in `zoom.ts:70`. Only needs UI wiring.

**Label rendering:** Already works. `loadOverlays()` processes all fields on
the sample. `processOverlays()` filters by `activePaths`. Labels from
`other_fields` render if enabled in the sidebar. The `t(state, x, y)` transform
maps normalized coordinates to canvas space correctly regardless of zoom/pan.

## What We Don't Build

- No label clipping/re-normalization — viewport transform handles visual
  clipping
- No new overlay rendering code — existing pipeline works
- No server-side image cropping — client-side canvas scale/pan
- No new PatchesView subclass — `ToTiles` produces standard `PatchesView`
- No label sync concerns — labels stay in original coordinates

## Files to Modify

| File                                 | Change                                 |
| ------------------------------------ | -------------------------------------- |
| `fiftyone/core/collections.py`       | Add `to_tiles()` method                |
| `fiftyone/core/stages.py`            | Add `ToTiles` ViewStage class          |
| `fiftyone/core/patches.py`           | Tile grid generation helper            |
| Grid action bar component (frontend) | Add `zoomPad` slider control           |
| Frontend state management            | Wire slider to `state.options.zoomPad` |

## User Experience

1. `view = dataset.to_tiles(tile_size=(640, 640), overlap=64)`
2. Grid shows tiles as cropped patches with annotations visible within each
   tile
3. User adjusts `zoomPad` slider to 0 for edge-to-edge tiles or higher for
   context
4. Clicking a tile opens full image zoomed to that region
5. De-zooming reveals the whole image with all annotations intact

## Revision (2026-09): tiles computed in the database

The first implementation wrote a temporary `tile_regions` field to every sample
of the source dataset, built the view with `make_patches_dataset()`, and
deleted the field again. That modified the source (every sample's
`last_modified_at`, and any user field named `tile_regions`), made concurrent
tiles views of one dataset silently mix their tiles, and broke tagging of tile
labels (the patches view synced them back to the deleted field). Tiles also
kept every label of their image.

The stage now works as follows:

- **Nothing is written to the source.** `make_tiles_dataset()` in
  `fiftyone/core/tiles.py` computes each image's tiles inside the aggregation
  that builds the tiles dataset (from `metadata.width/height`), mirroring
  upstream's `make_patches_dataset()` pipeline. It is also ~7x faster (84,000
  tiles of 2,000 4000x3000 images: 4.7s vs 32.6s)
- **Shift edge mode.** Each row/column uses the minimum number of tiles; the
  last one is shifted back to end at the image edge (its overlap with the
  previous tile grows). Where the image is smaller than the tile, tiles are
  clamped to the image size (no padding). `min_coverage` was removed; saved
  views that still carry it load with it ignored
- **Per-tile labels.** Top-level `Detections`/`Polylines`/`Keypoints` fields
  (and single-label variants) keep only the labels that touch the tile, in
  full-image coordinates. `min_label_coverage` optionally requires a minimum
  fraction of each label's area (bounding box area for polylines and keypoints)
  inside the tile. Image-level labels are copied as-is
- **View-local edits.** `TilesView` (a `PatchesView` subclass) syncs nothing
  back to the source, so tags and label edits stay in the view
- The App recognizes `fiftyone.core.tiles.TilesView` as a patches view
  (`PATCH_VIEWS` in `state/src/recoil/view.ts`); `tiles_tests.py` checks that
  the App's hardcoded names match the Python ones

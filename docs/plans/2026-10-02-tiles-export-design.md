# Tiles Export Design

## Problem

A tiles view (`fiftyone/core/tiles.py`) previews how a dataset looks when
tiled: each sample is a region of a full image, whose labels stay in full-image
coordinates. Exporting it with the built-in exporters produces broken data:
every tile points at its full source image, labels are unclipped, and
path-keyed formats (YOLO, VOC, KITTI, FiftyOne JSON,
ImageSegmentationDirectory) keep only the last tile of each image. Training on
tiles needs real crops with labels clipped to them and re-normalized.

## Solution

Two layers, both in fork-owned files so that no upstream exporter changes:

- **Materialize** (`fiftyone.utils.tiles.materialize_tiles()`,
  `TilesView.materialize()`): writes each tile as an image cropped from its
  source and builds a regular image dataset whose labels are clipped to the
  tiles. Every exporter then works unchanged, in every format, including the
  fork's own exporter changes.
- **Export** (`fiftyone.utils.tiles.export_tiles()`, `TilesView.export()`):
  materializes into a temporary directory and exports from there with
  `export_media="move"`, the way FiftyOne exports clips views. Because the
  tiles are new files, `export_media` can only be `True` or `"move"`.

Rejected: cropping on the fly inside `export_samples()`. It would modify
upstream's exporter code, only work with `export_media=True` (no symlink, move,
or manifest), name crops `000001.jpg`, and could not serve exporters that read
whole samples or collections (CSV, GeoJSON, FiftyOneDataset).

## Label transforms (`clip_label()`)

Labels are clipped in pixels of the full image, then re-normalized to the tile.
Every clipped label gets a new ID, since one label can be in several
overlapping tiles.

| Label                 | Transform                                                                                                                                                                                                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Detection             | Box intersected with the tile; dropped if no area is left. Instance masks are cropped to the visible part, at their own resolution                                                                                                                                                 |
| Polyline, filled      | Clipped as polygons; a shape that the tile cuts into several parts becomes one shape per part (see below). With `join_polygon_parts=True`, Sutherland-Hodgman keeps it one shape, its parts joined along the tile border by zero-area edges. Shapes inside the tile are kept as-is |
| Polyline, not filled  | Liang-Barsky line clipping; a line can split into several shapes. A closed outline that is split becomes open                                                                                                                                                                      |
| Keypoint              | Points outside the tile become NaN (like hidden points), keeping the order of the points for skeletons. Dropped if no point is visible                                                                                                                                             |
| Segmentation, Heatmap | Cropped to the tile at the array's own resolution                                                                                                                                                                                                                                  |
| Other labels, fields  | Copied as-is (classifications, scalars, ...)                                                                                                                                                                                                                                       |

## Polygon parts

By default, a filled shape that a tile cuts into several parts becomes one
shape per part, so YOLO writes one row (instance) per part, and COCO one
annotation with several polygons. This uses the Weiler-Atherton algorithm for a
rectangle (`_clip_polygon_parts()`): the chains of the polygon's boundary
inside the tile are linked, from each exit to the next entry along the tile's
border, into parts.

- Weiler-Atherton needs the polygon's interior on the same side of all its
  edges. The eta fork bridges holes to their outer boundary but runs them in
  the same direction as it (a bridged ring's signed area is the outer's plus
  the hole's), so `_orient_loops()` first decomposes each shape into the
  properly nested loops between its repeated vertices and reverses the ones
  that run the wrong way. This keeps every edge, so the shape is unchanged
  under the even-odd rule
- Where the tile's border cuts a zero-width seam, an exit and an entry share
  the same point. They are ordered as in the polygon shrunk by an infinitesimal
  amount, which keeps a hole inside the tile a hole of the part that surrounds
  it
- Shapes that cross themselves cannot be split: when the parts' total area
  differs from the Sutherland-Hodgman clipping's, the shape keeps the joined
  clipping

Tested against even-odd containment on random star polygons and on 18,000
random tiles of masks with holes and one-pixel-thin parts, in both modes.

Edges use the same tolerance as the tiles view's label filtering, so float
rounding never keeps a sliver or drops a label that touches an edge. Polygon
clipping uses no shapely: it is an optional dependency, and it rejects the
self-touching rings that the eta fork's hole bridging produces.

Masks and maps stored on disk are cropped as stored (no value conversion) and
written to `<output_dir>/fields/<field>/`; in-memory ones stay in memory.

The tiles view's `min_label_coverage` already is a visibility threshold: a
label that the view keeps in a tile has at least that fraction of its area
visible after clipping.

## Images

- Cropped with PIL, which keeps the source's mode and bit depth, in the
  orientation that the metadata describes. FiftyOne's metadata follows EXIF
  orientation for JPEGs but not for PNGs, so the image is EXIF-transposed only
  when that matches the metadata; otherwise it raises, asking to recompute the
  metadata
- PIL's decompression bomb limit is lifted while reading, since tiling is for
  large images
- Written in the source's format by default (`image_format` to change it; JPEG
  at quality 95), without EXIF orientation, as `<stem>_tile_<row>_<col><ext>`;
  `rel_dir` keeps the source directories
- Images smaller than a tile are not padded
- Each sample gets a `tile` field with the source `sample_id` and `filepath`,
  the tile's `row`/`col`, and its `x`, `y`, `width`, `height` in source pixels,
  to map predictions back to the full images
- Classes, mask targets, and skeletons are copied from the source dataset

## Exporters

Every image format has a round-trip test (`ExportFormatsTests` in
`tests/unittests/tiles_export_tests.py`): a tiles view is exported, read back
with FiftyOne's importer, and compared with the materialized tiles, within each
format's precision (exact for float formats, a pixel for formats that round).
Covered: YOLOv4/v5 (boxes, polygons, `use_masks=True`, splits, empty tiles),
COCO (boxes, masks, polygons, keypoints), VOC, KITTI, CVAT image, FiftyOne
image detection and image labels, BDD, image segmentation directory, the
classification formats, image and media directories, CSV, GeoJSON, FiftyOne
dataset and legacy dataset, and the TFRecords formats (with TensorFlow
installed). Notes:

- **YOLOv4/v5**: pass `classes` when exporting splits; tiles may leave a split
  without some classes, which would otherwise change the class indices.
  `export(empty_tiles=...)` decides what happens to tiles without labels in
  `label_field` (none of their image's labels touch them, or clipping removed
  them all): `"keep"` (default) exports them with empty label files (also when
  their source had no labels at all), `"keep_without_labels"` exports their
  images without label files, and `"skip"` leaves them out. The export logs how
  many tiles each choice affected.

    Only `"skip"` keeps them out of training: every trainer checked trains on
    images without a label file as backgrounds, exactly like images with an
    empty one. Checked in the code of Ultralytics' YOLOv3 (2018-2020), YOLOv5
    (v1.0 of June 2020 through 7.0.14), `ultralytics` 8.0.0 through 8.4.171,
    and AlexeyAB's Darknet (which also logs the missing file to `bad.list`).
    Running `ultralytics` 8.4.171's training dataloader on 2 labeled images, 1
    with an empty label file, and 2 without one fed all 5 images in every
    epoch, and a batch of the 3 backgrounds alone has a classification loss
    (0.23) and gradients, with no box loss. The one exception found is YOLOv5's
    opt-in `--image-weights`, which samples images by their label counts and so
    never samples backgrounds, with or without a label file

- **COCO**: cropped instance masks become polygons through the eta fork's
  hole-aware conversion; NaN keypoints are written as `(0, 0, 0)`, and read
  back as NaN. Importing polygon segmentations with `use_polylines=True`
  crashed on any COCO dataset, because a fork change to
  `_get_polygons_for_segmentation()` expected point pairs where COCO JSON has
  flat lists; fixed
- **VOC, KITTI**: detections that a tile cuts get `truncated`: VOC's flag
  becomes 1, KITTI's fraction accounts for the part outside the tile
  (`1 - (1 - t) * visible`). Existing values of other types are updated the
  same way by `clip_label()`; missing ones are only added for these formats
- **CVAT image**: crashed on NaN keypoints (also for hidden keypoints, without
  tiles); hidden points are now omitted, since CVAT points cannot be hidden,
  and keypoints without visible points are skipped. Instance masks are dropped
- **FiftyOne image labels**: instance masks cannot be read back (an eta bug,
  also without tiles: masks are written as lists but read as base64 strings),
  so the test exports boxes, polylines, and keypoints
- **ImageSegmentationDirectory**: needs the cropped segmentation masks, which
  materialization provides
- **TFRecords**: image filenames are not kept, so the test matches tiles by
  order
- **FiftyOneDataset**: keeps the `tile` provenance field

## Export tiles operator

The tiles plugin's "Export tiles" grid action (on tiles views) exports the
current tiles view from the App: a format (YOLOv5, YOLOv4, COCO, VOC, KITTI,
CVAT image, FiftyOne image detection, image segmentation directory, image
classification directory tree, FiftyOne dataset, or images only), a label field
of a type that the format exports, an export directory, what to do with
negative tiles (without labels), `join_polygon_parts` for polyline fields, the
YOLOv5 split, and whether to delete the directory first.

The choices for negative tiles depend on the format: the formats with a label
file per image (YOLOv5, YOLOv4, KITTI) offer all three (exported with empty
label files, exported without label files, not exported), while the other
formats with labels only keep or skip them, since they have no per-image label
files. YOLOv4 lists the tiles kept without label files in `images.txt` too.
Each set of choices has its own form parameter, so that a choice made for one
format does not carry over to another; formats without labels have none.

For YOLO and COCO, the classes come from the whole source dataset, so that
class indices stay the same across the splits and tilings exported. A summary
shows the number of tiles and images, the tiles without labels, and the
classes. The export runs immediately or as a delegated operation, and reports
its progress in two phases (cropping, then writing).

"""
Tiles utilities.

Turns a tiles view (:class:`fiftyone.core.tiles.TilesView`) into real data:
each tile becomes an image cropped from its source image, whose labels are
clipped to the tile and re-normalized to it.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from collections import OrderedDict
import logging
import math
import os
import re

import cv2
from PIL import Image, ImageOps

import eta.core.utils as etau

import fiftyone.core.dataset as fod
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.metadata as fome
import fiftyone.core.odm as foo
import fiftyone.core.sample as fos
import fiftyone.core.tiles as fot
import fiftyone.core.utils as fou
import fiftyone.utils.image as foui

logger = logging.getLogger(__name__)

# Fields of tiles views that only describe the view itself
_VIEW_FIELDS = {fot.TILE_FIELD, "sample_id"}

# Fields that every tile sample sets itself
_SAMPLE_FIELDS = {"id", "filepath", "metadata", "tags"}

_TILE_LABEL_PATTERN = re.compile(r"^tile_(\d+)_(\d+)$")

# Source images decoded at once; tiles views list the tiles of each image
# together, so each image is usually decoded once
_IMAGE_CACHE_SIZE = 2

_JPEG_EXTS = (".jpg", ".jpeg")
_JPEG_QUALITY = 95


def materialize_tiles(
    tiles_view,
    output_dir,
    fields=None,
    rel_dir=None,
    image_format=None,
    tile_field="tile",
    name=None,
    persistent=False,
    progress=None,
):
    """Creates a dataset that contains the tiles of a tiles view as images.

    Each tile becomes a sample whose image is the tile cropped from its
    source image, written to ``output_dir``. Its spatial labels are clipped
    to the tile and re-normalized to it, as described in
    :meth:`clip_label`; labels with no part left inside the tile are
    dropped. Other fields, including image-level labels such as
    classifications, are copied as-is.

    The ``tile_field`` of each sample records the tile's source: the
    ``sample_id`` and ``filepath`` of its source image, its ``row`` and
    ``col``, and its ``x``, ``y``, ``width``, and ``height`` in pixels of
    the source image.

    Images are cropped in the orientation that their metadata describes,
    which accounts for EXIF orientation, and are written without EXIF
    orientation.

    Args:
        tiles_view: a :class:`fiftyone.core.tiles.TilesView`
        output_dir: the directory in which to write the tile images, and the
            masks and heatmaps of labels that store them on disk
        fields (None): a field or list of fields of the view to include. By
            default, all fields are included
        rel_dir (None): an optional relative directory to strip from each
            source filepath to generate a unique path for its tiles in
            ``output_dir``. By default, tiles are named after the basename
            of their source image, with ``-<count>`` appended when needed to
            make them unique
        image_format (None): an optional image format, such as ``".png"``,
            in which to write the tile images. By default, the format of each
            source image is used
        tile_field ("tile"): the name of the field in which to record the
            source of each tile
        name (None): a name for the dataset
        persistent (False): whether the dataset should persist in the
            database after the session terminates
        progress (None): whether to render a progress bar (True/False), use
            the default value ``fiftyone.config.show_progress_bars`` (None),
            or a progress callback function to invoke instead

    Returns:
        a :class:`fiftyone.core.dataset.Dataset`
    """
    if not isinstance(tiles_view, fot.TilesView):
        raise ValueError(
            "Expected a %s, but found %s" % (fot.TilesView, type(tiles_view))
        )

    fields = _parse_fields(tiles_view, fields, tile_field)
    src_schema = tiles_view.get_field_schema()

    dataset = fod.Dataset(name=name, persistent=persistent)
    dataset.media_type = "image"
    dataset.add_sample_field(
        tile_field,
        fof.EmbeddedDocumentField,
        embedded_doc_type=foo.DynamicEmbeddedDocument,
    )
    if fields:
        dataset._sample_doc_cls.merge_field_schema(
            {f: src_schema[f] for f in fields}
        )

    _copy_label_settings(tiles_view._source_collection, dataset, fields)

    view = tiles_view.select_fields(fields + [fot.TILE_FIELD, "sample_id"])
    samples = _iter_tile_samples(
        view, output_dir, fields, rel_dir, image_format, tile_field
    )
    dataset.add_samples(samples, num_samples=len(view), progress=progress)

    return dataset


def export_tiles(
    tiles_view,
    export_dir=None,
    dataset_type=None,
    data_path=None,
    labels_path=None,
    export_media=None,
    rel_dir=None,
    dataset_exporter=None,
    label_field=None,
    frame_labels_field=None,
    overwrite=False,
    progress=None,
    **kwargs,
):
    """Exports the tiles of a tiles view as images with clipped labels.

    The tiles are first materialized via :meth:`materialize_tiles` into a
    temporary directory, from which the exporter moves them to their
    destination. This works with any exporter of image datasets, in any
    format: see :meth:`fiftyone.core.collections.SampleCollection.export`
    for the arguments.

    Because the tile images are new files, ``export_media`` can only be
    ``True`` or ``"move"``. To export tiles with other media options, such
    as symlinks, call :meth:`materialize_tiles` and export the resulting
    dataset.

    Args:
        tiles_view: a :class:`fiftyone.core.tiles.TilesView`
        export_dir (None): the directory to which to export the samples
        dataset_type (None): the :class:`fiftyone.types.Dataset` type to
            write
        data_path (None): an optional parameter that enables explicit
            control over the location of the exported media
        labels_path (None): an optional parameter that enables explicit
            control over the location of the exported labels
        export_media (None): ``None``, ``True``, or ``"move"``, which all
            write the tile images to the export location
        rel_dir (None): an optional relative directory to strip from each
            source filepath to generate a unique identifier for its tiles
        dataset_exporter (None): a
            :class:`fiftyone.utils.data.exporters.DatasetExporter` to use to
            export the samples
        label_field (None): the label field(s) to export
        frame_labels_field (None): not applicable to tiles, which are images
        overwrite (False): whether to delete existing directories before
            performing the export
        progress (None): whether to render a progress bar (True/False), use
            the default value ``fiftyone.config.show_progress_bars`` (None),
            or a progress callback function to invoke instead
        **kwargs: optional keyword arguments to pass to the dataset
            exporter's constructor
    """
    if export_media not in (None, True, "move"):
        raise ValueError(
            "Tiles are exported as new images, so `export_media` must be "
            "True or 'move', but found %r. To export tiles with other media "
            "options, call `materialize()` on the tiles view and export the "
            "resulting dataset" % (export_media,)
        )

    if dataset_exporter is not None:
        exporter_media = getattr(dataset_exporter, "export_media", True)
        if exporter_media not in (None, True, "move"):
            raise ValueError(
                "Tiles are exported as new images, so the exporter's "
                "`export_media` must be True or 'move', but found %r"
                % (exporter_media,)
            )

        if hasattr(dataset_exporter, "export_media"):
            dataset_exporter.export_media = "move"
    elif export_dir is None and data_path is None:
        raise ValueError(
            "Tiles are exported as new images, so `export_dir` or "
            "`data_path` must be provided"
        )

    tmp_dir = etau.make_temp_dir()
    dataset = None
    try:
        dataset = materialize_tiles(
            tiles_view, tmp_dir, rel_dir=rel_dir, progress=progress
        )

        # The tile images are temporary, so the exporter moves them rather
        # than copying them
        if dataset_exporter is None:
            kwargs["export_media"] = "move"

        dataset.export(
            export_dir=export_dir,
            dataset_type=dataset_type,
            data_path=data_path,
            labels_path=labels_path,
            rel_dir=tmp_dir if rel_dir is not None else None,
            dataset_exporter=dataset_exporter,
            label_field=label_field,
            frame_labels_field=frame_labels_field,
            overwrite=overwrite,
            progress=progress,
            **kwargs,
        )
    finally:
        if dataset is not None:
            dataset.delete()

        etau.delete_dir(tmp_dir)


def clip_label(label, tile, frame_size, mask_dir=None):
    """Clips a label to a tile of its image and re-normalizes it to the
    tile.

    The label must be in normalized coordinates of the full image:

    -   :class:`fiftyone.core.labels.Detection`: the bounding box is
        intersected with the tile, and the instance mask, if any, is cropped
        accordingly
    -   :class:`fiftyone.core.labels.Polyline`: filled shapes are clipped as
        polygons, along the tile's borders where they leave it. Other shapes
        are clipped as lines, which may split them into several shapes; a
        closed shape that is split becomes open
    -   :class:`fiftyone.core.labels.Keypoint`: points outside the tile
        become ``NaN``, like hidden points, which keeps the order of the
        points
    -   :class:`fiftyone.core.labels.Segmentation` and
        :class:`fiftyone.core.labels.Heatmap`: the mask or map is cropped to
        the tile, at its own resolution
    -   Lists of the above labels are clipped label by label

    Other labels, such as classifications, are returned as copies.

    Args:
        label: a :class:`fiftyone.core.labels.Label`
        tile: the ``(x, y, width, height)`` of the tile, in pixels
        frame_size: the ``(width, height)`` of the image, in pixels
        mask_dir (None): a directory in which to write the cropped masks and
            maps of labels that store them on disk. By default, cropped masks
            and maps are stored in memory

    Returns:
        a new :class:`fiftyone.core.labels.Label`, or ``None`` if no part of
        the label lies inside the tile
    """
    if isinstance(label, (fol.Detections, fol.Polylines, fol.Keypoints)):
        list_field = label._LABEL_LIST_FIELD
        clipped = []
        for _label in label[list_field]:
            _label = clip_label(_label, tile, frame_size, mask_dir=mask_dir)
            if _label is not None:
                clipped.append(_label)

        label = label.copy()
        label[list_field] = clipped
        return label

    if isinstance(label, fol.Detection):
        return _clip_detection(label, tile, frame_size, mask_dir)

    if isinstance(label, fol.Polyline):
        return _clip_polyline(label, tile, frame_size)

    if isinstance(label, fol.Keypoint):
        return _clip_keypoint(label, tile, frame_size)

    if isinstance(label, fol.Segmentation):
        return _crop_dense_label(label, "mask", tile, frame_size, mask_dir)

    if isinstance(label, fol.Heatmap):
        return _crop_dense_label(label, "map", tile, frame_size, mask_dir)

    return label.copy()


def _parse_fields(tiles_view, fields, tile_field):
    schema = tiles_view.get_field_schema()

    if fields is None:
        fields = [
            f
            for f in schema
            if f not in _VIEW_FIELDS
            and f not in _SAMPLE_FIELDS
            and f not in tiles_view._get_default_sample_fields()
        ]
    else:
        if etau.is_str(fields):
            fields = [fields]

        missing = [f for f in fields if f not in schema]
        if missing:
            raise ValueError("Tiles view has no field(s) %s" % missing)

        fields = [f for f in fields if f not in _SAMPLE_FIELDS]

        reserved = [f for f in fields if f in _VIEW_FIELDS]
        if reserved:
            raise ValueError(
                "Field(s) %s describe the tiles view and cannot be included"
                % reserved
            )

    if tile_field in fields or tile_field in _SAMPLE_FIELDS:
        raise ValueError(
            "The tiles view has a field '%s'; pass a different "
            "`tile_field`" % tile_field
        )

    return fields


def _copy_label_settings(src_collection, dataset, fields):
    dataset.classes = {
        f: c for f, c in src_collection.classes.items() if f in fields
    }
    dataset.default_classes = src_collection.default_classes
    dataset.mask_targets = {
        f: t for f, t in src_collection.mask_targets.items() if f in fields
    }
    dataset.default_mask_targets = src_collection.default_mask_targets
    dataset.skeletons = {
        f: s for f, s in src_collection.skeletons.items() if f in fields
    }
    dataset.default_skeleton = src_collection.default_skeleton


def _iter_tile_samples(
    view, output_dir, fields, rel_dir, image_format, tile_field
):
    filename_maker = fou.UniqueFilenameMaker(
        output_dir=output_dir, rel_dir=rel_dir, ignore_existing=True
    )
    images = _ImageCache(_IMAGE_CACHE_SIZE)

    for tile_sample in view.iter_samples():
        frame_size = (tile_sample.metadata.width, tile_sample.metadata.height)
        region = tile_sample[fot.TILE_FIELD]
        tile = _tile_box(region.bounding_box, frame_size)
        x, y, width, height = tile

        img = images.get(tile_sample.filepath, frame_size)
        outpath = filename_maker.get_output_path(
            _tile_path(tile_sample.filepath, region.label, image_format)
        )
        _write_image(img.crop((x, y, x + width, y + height)), outpath)

        row, col = _parse_row_col(region.label)
        sample = fos.Sample(
            filepath=outpath,
            metadata=fome.ImageMetadata.build_for(outpath),
            tags=list(tile_sample.tags),
        )
        sample[tile_field] = foo.DynamicEmbeddedDocument(
            sample_id=str(tile_sample.sample_id),
            filepath=tile_sample.filepath,
            row=row,
            col=col,
            x=x,
            y=y,
            width=width,
            height=height,
        )

        for field in fields:
            value = tile_sample[field]
            if isinstance(value, fol.Label):
                mask_dir = os.path.join(output_dir, "fields", field)
                value = clip_label(value, tile, frame_size, mask_dir=mask_dir)

            sample[field] = value

        yield sample


def _tile_box(bounding_box, frame_size):
    # Tiles are whole pixels, which their normalized boxes round-trip
    width, height = frame_size
    x, y, w, h = bounding_box
    return (
        int(round(x * width)),
        int(round(y * height)),
        int(round(w * width)),
        int(round(h * height)),
    )


def _tile_path(src_path, tile_label, image_format):
    stem, ext = os.path.splitext(os.path.basename(src_path))
    if image_format is not None:
        ext = image_format

    filename = "%s_%s%s" % (stem, tile_label, ext)
    return os.path.join(os.path.dirname(src_path), filename)


def _parse_row_col(tile_label):
    match = _TILE_LABEL_PATTERN.match(tile_label or "")
    if match is None:
        return None, None

    return int(match.group(1)), int(match.group(2))


class _ImageCache(object):
    """A cache of the most recently decoded source images."""

    def __init__(self, size):
        self._size = size
        self._images = OrderedDict()

    def get(self, path, frame_size):
        img = self._images.pop(path, None)
        if img is None:
            img = _read_image(path, frame_size)

        self._images[path] = img
        while len(self._images) > self._size:
            self._images.popitem(last=False)

        return img


def _read_image(path, frame_size):
    # Tiling is meant for large images, so PIL's decompression bomb check,
    # which rejects images above ~179 megapixels, does not apply
    max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as img:
            img.load()
            oriented = ImageOps.exif_transpose(img)
    finally:
        Image.MAX_IMAGE_PIXELS = max_pixels

    # Tiles and labels are in the orientation that the metadata describes,
    # which follows the image's EXIF orientation when FiftyOne can read it
    # without decoding the image (as for JPEGs, but not PNGs)
    frame_size = tuple(frame_size)
    if oriented.size == frame_size:
        return oriented

    if img.size == frame_size:
        return img

    raise ValueError(
        "Image '%s' is %d x %d, but its metadata says %d x %d. Recompute its "
        "metadata via `compute_metadata(overwrite=True)` and create the "
        "tiles view again" % ((path,) + tuple(oriented.size) + frame_size)
    )


def _write_image(img, path):
    etau.ensure_basedir(path)

    if os.path.splitext(path)[1].lower() in _JPEG_EXTS:
        if img.mode not in ("L", "RGB", "CMYK"):
            img = img.convert("RGB")

        img.save(path, quality=_JPEG_QUALITY)
    else:
        img.save(path)


def _clip_detection(detection, tile, frame_size, mask_dir):
    box = detection.bounding_box
    if not _is_valid_box(box):
        return None

    img_w, img_h = frame_size
    tx, ty, tw, th = tile
    x1, y1 = box[0] * img_w, box[1] * img_h
    x2, y2 = x1 + box[2] * img_w, y1 + box[3] * img_h

    span_x = _clip_span(x1, x2, tx, tx + tw, img_w)
    span_y = _clip_span(y1, y2, ty, ty + th, img_h)
    if span_x is None or span_y is None:
        return None

    (cx1, cx2), (cy1, cy2) = span_x, span_y

    clipped = detection.copy()
    clipped.bounding_box = [
        (cx1 - tx) / tw,
        (cy1 - ty) / th,
        (cx2 - cx1) / tw,
        (cy2 - cy1) / th,
    ]

    if detection.has_mask and x2 > x1 and y2 > y1:
        # The mask spans the box, at its own resolution
        def crop(mask_w, mask_h):
            return (
                _crop_range(cy1 - y1, cy2 - y1, y2 - y1, mask_h),
                _crop_range(cx1 - x1, cx2 - x1, x2 - x1, mask_w),
            )

        _crop_mask(detection, clipped, "mask", crop, mask_dir)

    return clipped


def _is_valid_box(box):
    return (
        box is not None
        and len(box) == 4
        and all(v is not None and math.isfinite(v) for v in box)
    )


def _clip_span(start, end, tile_start, tile_end, img_len):
    """Clips the span ``[start, end]`` to the tile's ``[tile_start,
    tile_end]``, or returns None if the span has no part inside the tile.

    A span with length must overlap the tile with positive length; a
    degenerate span must intersect the tile. Edges are compared with the
    same tolerance as when filtering the labels of tiles views.
    """
    eps = fot._EPS * img_len
    clipped_start = max(start, tile_start)
    clipped_end = min(end, tile_end)

    if end - start > eps:
        if clipped_end - clipped_start <= eps:
            return None
    elif clipped_end - clipped_start < -eps:
        return None

    return clipped_start, max(clipped_end, clipped_start)


def _crop_range(start, end, length, size):
    """Converts the span ``[start, end]`` of a ``length`` that an array of
    ``size`` elements covers to a non-empty index range of the array.
    """
    i0 = min(max(int(round(start / length * size)), 0), size - 1)
    i1 = min(max(int(round(end / length * size)), i0 + 1), size)
    return i0, i1


def _crop_dense_label(label, attr, tile, frame_size, mask_dir):
    # Masks and maps span the image, at their own resolution
    img_w, img_h = frame_size
    tx, ty, tw, th = tile

    def crop(mask_w, mask_h):
        return (
            _crop_range(ty, ty + th, img_h, mask_h),
            _crop_range(tx, tx + tw, img_w, mask_w),
        )

    cropped = label.copy()
    _crop_mask(label, cropped, attr, crop, mask_dir)
    return cropped


def _crop_mask(label, cropped, attr, crop, mask_dir):
    """Crops the mask or map ``attr`` of ``label`` into ``cropped``.

    In-memory masks are cropped in memory. Masks on disk are cropped as
    stored, so their values are not converted, and written to ``mask_dir``
    when one is provided.
    """
    path_attr = attr + "_path"
    path = label[path_attr]

    if path is None:
        mask = label[attr]
        if mask is None:
            return

        (r0, r1), (c0, c1) = crop(mask.shape[1], mask.shape[0])
        cropped[attr] = mask[r0:r1, c0:c1].copy()
        return

    # pylint: disable=no-member
    mask = foui.read(path, flag=cv2.IMREAD_UNCHANGED)
    (r0, r1), (c0, c1) = crop(mask.shape[1], mask.shape[0])
    mask = mask[r0:r1, c0:c1]

    if mask_dir is None:
        cropped[attr] = mask.copy()
        cropped[path_attr] = None
        return

    ext = os.path.splitext(path)[1] or ".png"
    outpath = os.path.join(mask_dir, cropped.id + ext)
    etau.ensure_basedir(outpath)
    foui.write(mask, outpath)
    cropped[path_attr] = outpath


def _clip_polyline(polyline, tile, frame_size):
    img_w, img_h = frame_size
    tx, ty, tw, th = tile
    rect = (tx, ty, tx + tw, ty + th)
    eps = fot._EPS * max(img_w, img_h)

    shapes = []
    was_split = False
    for shape in polyline.points or []:
        points = [
            (x * img_w, y * img_h)
            for x, y in shape
            if x is not None
            and y is not None
            and math.isfinite(x)
            and math.isfinite(y)
        ]
        if not points:
            continue

        if _is_inside(points, rect, eps):
            # Kept as-is, even if degenerate
            shapes.append([_clamp_point(p, rect) for p in points])
        elif polyline.filled:
            points = _clip_polygon(points, rect, eps)
            if len(points) >= 3 and abs(_polygon_area(points)) > eps * eps:
                shapes.append(points)
        else:
            pieces, split = _clip_line(points, rect, polyline.closed, eps)
            shapes.extend(pieces)
            was_split |= split

    if not shapes:
        return None

    clipped = polyline.copy()
    clipped.points = [
        [((x - tx) / tw, (y - ty) / th) for x, y in shape] for shape in shapes
    ]

    if was_split:
        # The shapes that were not split explicitly repeat their first point
        clipped.closed = False

    return clipped


def _clip_polygon(points, rect, eps):
    """Clips a polygon to a rectangle via the Sutherland-Hodgman algorithm.

    A concave polygon that the rectangle cuts into several parts stays one
    polygon, whose parts are joined by edges along the rectangle's border.
    """
    x0, y0, x1, y1 = rect
    edges = (
        (lambda p: p[0] >= x0, lambda p, q: _at_x(p, q, x0)),
        (lambda p: p[0] <= x1, lambda p, q: _at_x(p, q, x1)),
        (lambda p: p[1] >= y0, lambda p, q: _at_y(p, q, y0)),
        (lambda p: p[1] <= y1, lambda p, q: _at_y(p, q, y1)),
    )

    for inside, intersect in edges:
        if not points:
            break

        clipped = []
        prev = points[-1]
        for point in points:
            if inside(point):
                if not inside(prev):
                    clipped.append(intersect(prev, point))

                clipped.append(point)
            elif inside(prev):
                clipped.append(intersect(prev, point))

            prev = point

        points = clipped

    return _dedupe_points(points, eps, closed=True)


def _at_x(p, q, x):
    t = (x - p[0]) / (q[0] - p[0])
    return (x, p[1] + t * (q[1] - p[1]))


def _at_y(p, q, y):
    t = (y - p[1]) / (q[1] - p[1])
    return (p[0] + t * (q[0] - p[0]), y)


def _dedupe_points(points, eps, closed=False):
    deduped = []
    for point in points:
        if not deduped or not _same_point(point, deduped[-1], eps):
            deduped.append(point)

    if (
        closed
        and len(deduped) > 1
        and _same_point(deduped[0], deduped[-1], eps)
    ):
        deduped.pop()

    return deduped


def _same_point(p, q, eps):
    return abs(p[0] - q[0]) <= eps and abs(p[1] - q[1]) <= eps


def _polygon_area(points):
    area = 0.0
    for (xa, ya), (xb, yb) in zip(points, points[1:] + points[:1]):
        area += xa * yb - xb * ya

    return area / 2


def _clip_line(points, rect, closed, eps):
    """Clips a line, closed or not, to a rectangle.

    Returns:
        a tuple of the list of pieces of the line inside the rectangle, and
        whether the line was split
    """
    if closed:
        points = points + points[:1]

    pieces = []
    piece = []
    for p, q in zip(points, points[1:]):
        segment = _clip_segment(p, q, rect, eps)
        if segment is None:
            if piece:
                pieces.append(piece)
                piece = []

            continue

        start, end = segment
        if piece and _same_point(piece[-1], start, eps):
            piece.append(end)
        else:
            if piece:
                pieces.append(piece)

            piece = [start, end]

    if piece:
        pieces.append(piece)

    # A closed line may leave the rectangle and come back to its start
    if (
        closed
        and len(pieces) > 1
        and _same_point(pieces[-1][-1], pieces[0][0], eps)
    ):
        pieces[0] = pieces.pop() + pieces[0][1:]

    pieces = [_dedupe_points(p, eps) for p in pieces]
    pieces = [p for p in pieces if len(p) >= 2]
    return pieces, True


def _clip_segment(p, q, rect, eps):
    """Clips a segment to a rectangle, with a tolerance, via the
    Liang-Barsky algorithm.
    """
    x0, y0, x1, y1 = rect
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0

    for pk, qk in (
        (-dx, p[0] - (x0 - eps)),
        (dx, (x1 + eps) - p[0]),
        (-dy, p[1] - (y0 - eps)),
        (dy, (y1 + eps) - p[1]),
    ):
        if pk == 0:
            if qk < 0:
                return None
        elif pk < 0:
            t0 = max(t0, qk / pk)
        else:
            t1 = min(t1, qk / pk)

        if t0 > t1:
            return None

    start = (p[0] + t0 * dx, p[1] + t0 * dy)
    end = (p[0] + t1 * dx, p[1] + t1 * dy)
    return _clamp_point(start, rect), _clamp_point(end, rect)


def _is_inside(points, rect, eps):
    x0, y0, x1, y1 = rect
    return all(
        x0 - eps <= x <= x1 + eps and y0 - eps <= y <= y1 + eps
        for x, y in points
    )


def _clamp_point(point, rect):
    x0, y0, x1, y1 = rect
    return (min(max(point[0], x0), x1), min(max(point[1], y0), y1))


def _clip_keypoint(keypoint, tile, frame_size):
    img_w, img_h = frame_size
    tx, ty, tw, th = tile
    rect = (tx, ty, tx + tw, ty + th)
    eps = fot._EPS * max(img_w, img_h)
    nan = float("nan")

    points = []
    num_visible = 0
    for x, y in keypoint.points or []:
        if (
            x is None
            or y is None
            or not math.isfinite(x)
            or not math.isfinite(y)
        ):
            points.append((nan, nan))
            continue

        px, py = x * img_w, y * img_h
        if tx - eps <= px <= tx + tw + eps and ty - eps <= py <= ty + th + eps:
            px, py = _clamp_point((px, py), rect)
            points.append(((px - tx) / tw, (py - ty) / th))
            num_visible += 1
        else:
            points.append((nan, nan))

    if not num_visible:
        return None

    clipped = keypoint.copy()
    clipped.points = points
    return clipped

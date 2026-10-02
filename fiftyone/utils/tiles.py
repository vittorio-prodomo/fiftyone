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
import fnmatch
import logging
import math
import os
import re

import cv2
from PIL import Image, ImageOps

import eta.core.utils as etau

import fiftyone.core.dataset as fod
import fiftyone.core.expressions as foe
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.metadata as fome
import fiftyone.core.odm as foo
import fiftyone.core.sample as fos
import fiftyone.core.tiles as fot
import fiftyone.core.utils as fou
import fiftyone.utils.image as foui

logger = logging.getLogger(__name__)

F = foe.ViewField

# Fields of tiles views that only describe the view itself
_VIEW_FIELDS = {fot.TILE_FIELD, "sample_id"}

# Fields that every tile sample sets itself
_SAMPLE_FIELDS = {"id", "filepath", "metadata", "tags"}

_TILE_LABEL_PATTERN = re.compile(r"^tile_(\d+)_(\d+)$")

# Source images decoded at once; tiles views list the tiles of each image
# together, so each image is usually decoded once
_IMAGE_CACHE_SIZE = 2

_JPEG_EXTS = (".jpg", ".jpeg")

_EMPTY_TILES = ("keep", "keep_without_labels", "skip")
_JPEG_QUALITY = 95


def materialize_tiles(
    tiles_view,
    output_dir,
    fields=None,
    rel_dir=None,
    image_format=None,
    tile_field="tile",
    join_polygon_parts=False,
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
        join_polygon_parts (False): whether to keep a filled polyline shape
            that a tile cuts into several parts as one shape, whose parts are
            joined by zero-area edges along the tile's border, rather than
            one shape per part. See :meth:`clip_label`
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
        view,
        output_dir,
        fields,
        rel_dir,
        image_format,
        tile_field,
        join_polygon_parts,
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
    empty_tiles="keep",
    join_polygon_parts=False,
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

    Tiles may have no labels in ``label_field``, either because none of
    their image's labels touch them or because clipping removed them all.
    ``empty_tiles`` decides how such tiles are exported. In YOLO and similar
    formats, ``"keep"`` exports their images with empty label files, and
    ``"keep_without_labels"`` exports their images without label files.
    Note that trainers such as Ultralytics' (YOLOv3 since 2018, YOLOv5, and
    YOLOv8 and later) and Darknet use both as background images (negatives),
    so ``"skip"`` is the way to leave them out of training.

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
        empty_tiles ("keep"): what to do with the tiles that have no labels
            in ``label_field``:

            -   ``"keep"``: export them with empty labels. Tiles whose source
                had no labels at all get empty labels too, for lists of
                labels such as :class:`fiftyone.core.labels.Detections`
            -   ``"keep_without_labels"``: export them without labels
            -   ``"skip"``: do not export them

            Values other than ``"keep"`` require ``label_field``
        join_polygon_parts (False): whether to keep a filled polyline shape
            that a tile cuts into several parts as one shape, whose parts are
            joined by zero-area edges along the tile's border, rather than
            one shape per part. See :meth:`clip_label`
        **kwargs: optional keyword arguments to pass to the dataset
            exporter's constructor
    """
    if empty_tiles not in _EMPTY_TILES:
        raise ValueError(
            "`empty_tiles` must be one of %s, but found %r"
            % (_EMPTY_TILES, empty_tiles)
        )

    label_fields = _parse_label_fields(tiles_view, label_field)
    if empty_tiles != "keep" and not label_fields:
        raise ValueError(
            "`empty_tiles=%r` requires `label_field`, whose labels decide "
            "which tiles are empty" % empty_tiles
        )

    num_tiles = len(tiles_view)
    if empty_tiles == "skip":
        # Tiles without labels in the view have none once clipped either
        tiles_view = tiles_view.match(
            _has_labels_expr(tiles_view, label_fields)
        )

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
            tiles_view,
            tmp_dir,
            rel_dir=rel_dir,
            join_polygon_parts=join_polygon_parts,
            progress=progress,
        )

        export_view = dataset
        if label_fields:
            export_view = _handle_empty_tiles(
                dataset, label_fields, empty_tiles, num_tiles
            )

        # The tile images are temporary, so the exporter moves them rather
        # than copying them
        if dataset_exporter is None:
            kwargs["export_media"] = "move"

        export_view.export(
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


def clip_label(
    label, tile, frame_size, mask_dir=None, join_polygon_parts=False
):
    """Clips a label to a tile of its image and re-normalizes it to the
    tile.

    The label must be in normalized coordinates of the full image:

    -   :class:`fiftyone.core.labels.Detection`: the bounding box is
        intersected with the tile, and the instance mask, if any, is cropped
        accordingly
    -   :class:`fiftyone.core.labels.Polyline`: filled shapes are clipped as
        polygons, along the tile's borders where they leave it. A shape that
        the tile cuts into several parts becomes one shape per part, unless
        ``join_polygon_parts`` is True. Other shapes are clipped as lines,
        which may also split them into several shapes; a closed shape that
        is split becomes open
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
        join_polygon_parts (False): whether to keep a filled shape that the
            tile cuts into several parts as one shape, whose parts are joined
            by zero-area edges along the tile's border (True), rather than
            one shape per part (False)

    Returns:
        a new :class:`fiftyone.core.labels.Label`, or ``None`` if no part of
        the label lies inside the tile
    """
    if isinstance(label, (fol.Detections, fol.Polylines, fol.Keypoints)):
        list_field = label._LABEL_LIST_FIELD
        clipped = []
        for _label in label[list_field]:
            _label = clip_label(
                _label,
                tile,
                frame_size,
                mask_dir=mask_dir,
                join_polygon_parts=join_polygon_parts,
            )
            if _label is not None:
                clipped.append(_label)

        label = label.copy()
        label[list_field] = clipped
        return label

    if isinstance(label, fol.Detection):
        return _clip_detection(label, tile, frame_size, mask_dir)

    if isinstance(label, fol.Polyline):
        return _clip_polyline(label, tile, frame_size, join_polygon_parts)

    if isinstance(label, fol.Keypoint):
        return _clip_keypoint(label, tile, frame_size)

    if isinstance(label, fol.Segmentation):
        return _crop_dense_label(label, "mask", tile, frame_size, mask_dir)

    if isinstance(label, fol.Heatmap):
        return _crop_dense_label(label, "map", tile, frame_size, mask_dir)

    return label.copy()


def _handle_empty_tiles(dataset, label_fields, empty_tiles, num_tiles):
    """Applies ``empty_tiles`` to the tiles without labels in
    ``label_fields`` of a materialized tiles dataset, and returns the view to
    export.
    """
    has_labels = _has_labels_expr(dataset, label_fields)
    empty = dataset.match(~has_labels)
    num_empty = len(empty)

    if empty_tiles == "skip":
        num_skipped = num_tiles - len(dataset) + num_empty
        if num_skipped:
            logger.info(
                "Skipping %d of %d tile(s) without labels in %s",
                num_skipped,
                num_tiles,
                label_fields,
            )

        return dataset.match(has_labels)

    if not num_empty:
        return dataset

    if empty_tiles == "keep_without_labels":
        for field in label_fields:
            empty.set_values(field, [None] * num_empty)

        logger.info(
            "Exporting %d of %d tile(s) without labels in %s, as background "
            "images without labels (no label files, in YOLO and similar "
            "formats)",
            num_empty,
            num_tiles,
            label_fields,
        )
        return dataset

    # Tiles whose source had no labels at all also get empty labels, where
    # the label type has an empty value (lists of labels)
    for field in label_fields:
        doc_type = getattr(dataset.get_field(field), "document_type", None)
        if getattr(doc_type, "_LABEL_LIST_FIELD", None) is None:
            continue

        missing = empty.match(F(field) == None)
        missing.set_values(field, [doc_type() for _ in range(len(missing))])

    logger.info(
        "Exporting %d of %d tile(s) without labels in %s, as background "
        "images with empty labels (empty label files, in YOLO and similar "
        "formats). Pass `empty_tiles='skip'` to leave them out",
        num_empty,
        num_tiles,
        label_fields,
    )
    return dataset


def _parse_label_fields(collection, label_field):
    """The fields that ``label_field`` exports, as
    :meth:`fiftyone.core.collections.SampleCollection.export` accepts it.
    """
    if label_field is None:
        return []

    if isinstance(label_field, dict):
        return list(label_field.keys())

    if etau.is_str(label_field):
        if any(c in label_field for c in "*?["):
            schema = collection.get_field_schema()
            return [f for f in schema if fnmatch.fnmatch(f, label_field)]

        return [label_field]

    return list(label_field)


def _has_labels_expr(collection, label_fields):
    """An expression that matches the samples that have labels in any of the
    given fields.
    """
    exprs = []
    for field in label_fields:
        doc_type = getattr(collection.get_field(field), "document_type", None)
        list_field = getattr(doc_type, "_LABEL_LIST_FIELD", None)
        if list_field is not None:
            exprs.append(F(field + "." + list_field).length() > 0)
        else:
            exprs.append(F(field) != None)

    return F.any(exprs)


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
    view,
    output_dir,
    fields,
    rel_dir,
    image_format,
    tile_field,
    join_polygon_parts,
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
                value = clip_label(
                    value,
                    tile,
                    frame_size,
                    mask_dir=mask_dir,
                    join_polygon_parts=join_polygon_parts,
                )

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


def _clip_polyline(polyline, tile, frame_size, join_polygon_parts):
    img_w, img_h = frame_size
    tx, ty, tw, th = tile
    rect = (tx, ty, tx + tw, ty + th)
    eps = fot._EPS * max(img_w, img_h)

    shapes = []
    closed_shapes = []
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
            if polyline.closed and not polyline.filled:
                closed_shapes.append(len(shapes))

            shapes.append([_clamp_point(p, rect) for p in points])
        elif polyline.filled:
            points = _orient_loops(points)
            joined = _clip_polygon(points, rect, eps)
            if join_polygon_parts:
                parts = [joined]
            else:
                parts = _clip_polygon_parts(points, rect, eps)

                # Splitting is only defined for (weakly) simple polygons;
                # self-intersecting ones keep their joined clipping
                area = sum(abs(_polygon_area(part)) for part in parts)
                expected = abs(_polygon_area(joined)) if joined else 0.0
                if abs(area - expected) > 1e-6 * max(expected, 1.0):
                    parts = [joined]

            shapes.extend(
                part
                for part in parts
                if len(part) >= 3 and abs(_polygon_area(part)) > eps * eps
            )
        else:
            pieces, split = _clip_line(points, rect, polyline.closed, eps)
            shapes.extend(pieces)
            was_split |= split

    if not shapes:
        return None

    if was_split:
        # The label becomes open, so its closed shapes that were not split
        # explicitly repeat their first point
        for i in closed_shapes:
            shapes[i] = shapes[i] + shapes[i][:1]

    clipped = polyline.copy()
    clipped.points = [
        [((x - tx) / tw, (y - ty) / th) for x, y in shape] for shape in shapes
    ]

    if was_split:
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


def _orient_loops(points):
    """Orients the loops of a polygon that start and end at a repeated
    vertex, such as holes bridged to their outer boundary by zero-width
    seams, so that the polygon is weakly simple: loops inside their
    enclosing loop (holes) run opposite to it, and other loops along it.

    This keeps the polygon's edges, so it does not change the polygon under
    the even-odd rule, but makes its interior lie on the same side of all
    its edges, which splitting it requires. The eta fork's hole bridging,
    for example, runs holes along their outer boundary.
    """
    # Decompose the polygon into properly nested loops
    loops = []
    path = []
    on_path = {}
    for i, point in enumerate(points):
        key = tuple(point)
        k = on_path.get(key)
        if k is None:
            on_path[key] = len(path)
            path.append(i)
            continue

        loops.append((path[k], i))
        for j in path[k + 1 :]:
            del on_path[tuple(points[j])]

        del path[k + 1 :]

    if not loops:
        return list(points)

    root = _Loop(0, len(points))
    stack = [root]
    for start, end in sorted(loops, key=lambda l: (l[0], -l[1])):
        while not stack[-1].contains(start, end):
            stack.pop()

        loop = _Loop(start, end)
        stack[-1].children.append(loop)
        stack.append(loop)

    root.orient(points, None, None)
    return root.expand(points)


class _Loop(object):
    """A loop of a polygon, from its vertex ``start`` to the repeated vertex
    ``end``, or the whole polygon.
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.children = []
        self.reverse = False

    def contains(self, start, end):
        return self.start <= start and end <= self.end

    def own_indices(self):
        """The loop's vertices, without those of the loops nested in it."""
        indices = []
        i = self.start
        children = iter(sorted(self.children, key=lambda c: c.start))
        child = next(children, None)
        while i < self.end:
            indices.append(i)
            if child is not None and child.start == i:
                i = child.end + 1
                child = next(children, None)
            else:
                i += 1

        return indices

    def orient(self, points, outer, outer_positive):
        """Decides the orientation of this loop and the loops in it, given
        the nearest enclosing loop with an area and its orientation.
        """
        own = [points[i] for i in self.own_indices()]
        area = _polygon_area(own) if len(own) >= 3 else 0.0

        if area and outer is not None:
            is_hole = _is_inside_polygon(_interior_points(own, area), outer)
            positive = outer_positive != is_hole
            self.reverse = (area > 0) != positive
        elif area:
            positive = area > 0
        else:
            # Degenerate, such as a seam: its loops relate to its outer loop
            own, positive = outer, outer_positive

        for child in self.children:
            child.orient(points, own, positive)

    def expand(self, points):
        """The loop's vertices, from its start, with its nested loops."""
        indices = self.own_indices()
        if self.reverse:
            indices = indices[:1] + indices[:0:-1]

        children = {}
        for child in self.children:
            children.setdefault(child.start, []).append(child)

        expanded = []
        for i in indices:
            expanded.append(points[i])
            for child in children.get(i, []):
                expanded.extend(child.expand(points)[1:])
                expanded.append(points[i])

        return expanded


def _interior_points(polygon, area, num=3):
    """Points just inside a polygon, next to the midpoints of its longest
    edges, which avoids its vertices, which other polygons may share.
    """
    edges = sorted(
        zip(polygon, polygon[1:] + polygon[:1]),
        key=lambda e: (e[1][0] - e[0][0]) ** 2 + (e[1][1] - e[0][1]) ** 2,
        reverse=True,
    )

    # The interior is on the left of the edges of positive area polygons
    sign = 1.0 if area > 0 else -1.0
    points = []
    for (xa, ya), (xb, yb) in edges[:num]:
        length = math.hypot(xb - xa, yb - ya)
        if not length:
            continue

        offset = sign * 1e-6 * length
        points.append(
            (
                (xa + xb) / 2 - offset * (yb - ya) / length,
                (ya + yb) / 2 + offset * (xb - xa) / length,
            )
        )

    return points


def _is_inside_polygon(points, polygon):
    """Whether most of the points are inside the polygon."""
    num_inside = sum(_point_in_polygon(p, polygon) for p in points)
    return 2 * num_inside > len(points)


def _clip_polygon_parts(points, rect, eps):
    """Clips a polygon to a rectangle, as one polygon per part of it inside
    the rectangle.

    This is the Weiler-Atherton algorithm for a rectangle: it collects the
    chains of the polygon's boundary inside the rectangle, and links each
    chain's exit to the next chain's entry along the rectangle's border, in
    the polygon's orientation. Weakly simple polygons, such as those whose
    holes are bridged to their outer boundary by zero-width seams, are
    supported.
    """
    if len(points) < 3:
        return []

    if _polygon_area(points) < 0:
        points = points[::-1]

    # Start outside, so that no chain wraps around
    start = next(
        (i for i, p in enumerate(points) if not _is_inside([p], rect, eps)),
        None,
    )
    if start is None:
        return [points]

    points = points[start:] + points[:start]

    # Each chain is (points, entry crossing, exit crossing)
    chains = []
    chain = None
    for p, q in zip(points, points[1:] + points[:1]):
        direction = (q[0] - p[0], q[1] - p[1])
        q_inside = _is_inside([q], rect, eps)
        if chain is not None:
            # Inside, as the rectangle is convex
            if q_inside:
                chain[0].append(_clamp_point(q, rect))
                continue

            segment = _clip_segment(p, q, rect, eps)
            exit_ = segment[1] if segment else _clamp_point(p, rect)
            chain[0].append(exit_)
            chain.append(_crossing(exit_, direction, rect))
            chains.append(chain)
            chain = None
            continue

        segment = _clip_segment(p, q, rect, eps)
        if segment is None:
            continue

        entry, end = segment
        if q_inside:
            chain = [[entry, end], _crossing(entry, direction, rect)]
        elif not _same_point(entry, end, eps):
            chains.append(
                [
                    [entry, end],
                    _crossing(entry, direction, rect),
                    _crossing(end, direction, rect),
                ]
            )

    if not chains:
        # The polygon either contains the rectangle or misses it
        x0, y0, x1, y1 = rect
        if _point_in_polygon(((x0 + x1) / 2, (y0 + y1) / 2), points):
            return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]

        return []

    perimeter = 2 * ((rect[2] - rect[0]) + (rect[3] - rect[1]))

    def walk(exit_, entry):
        """How far ``entry`` is after ``exit_`` along the border, in the
        polygon's orientation, as a sortable ``(distance, order)``.
        """
        d = (entry[0] - exit_[0]) % perimeter
        if eps < d < perimeter - eps:
            return d, 0.0

        # The same point, such as where the border cuts a zero-width seam:
        # the crossings are ordered as in the polygon shrunk by an
        # infinitesimal amount, which moves each of them to the inner side
        # of its edge
        order = entry[1] - exit_[1]
        return (0.0 if order > 1e-12 else perimeter), order

    parts = []
    used = [False] * len(chains)
    for first in range(len(chains)):
        if used[first]:
            continue

        part = []
        i = first
        while True:
            used[i] = True
            part.extend(chains[i][0])

            exit_ = chains[i][2]
            i = min(
                range(len(chains)),
                key=lambda j: walk(exit_, chains[j][1]) + (used[j],),
            )
            distance = walk(exit_, chains[i][1])[0]
            part.extend(_corners_between(exit_[0], distance, rect, eps))

            if used[i]:
                break

        part = _remove_spikes(_dedupe_points(part, eps, closed=True), eps)
        parts.append(part)

    return parts


def _crossing(point, direction, rect):
    """Where the polygon's edge with ``direction`` crosses the rectangle's
    border at ``point``: its border position, and how far an infinitesimal
    shift of the edge to its inner (left) side moves it along the border.
    """
    position, side = _border_position(point, rect)
    tangent = ((1, 0), (0, 1), (-1, 0), (0, -1))[side]
    dx, dy = direction
    norm = math.hypot(dx, dy) or 1.0
    return position, (-dy * tangent[0] + dx * tangent[1]) / norm


def _border_position(point, rect):
    """The position of a point on the rectangle's border, measured along it
    from ``(x0, y0)`` in the orientation of polygons with positive area, and
    the index of its side.
    """
    x0, y0, x1, y1 = rect
    x, y = point
    w, h = x1 - x0, y1 - y0
    dists = (abs(y - y0), abs(x - x1), abs(y - y1), abs(x - x0))
    side = dists.index(min(dists))
    if side == 0:
        return x - x0, side

    if side == 1:
        return w + (y - y0), side

    if side == 2:
        return w + h + (x1 - x), side

    return 2 * w + h + (y1 - y), side


def _corners_between(s_from, distance, rect, eps):
    """The corners of the rectangle that lie strictly within ``distance``
    after border position ``s_from``, in order.
    """
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    perimeter = 2 * (w + h)
    corners = (
        (0, (x0, y0)),
        (w, (x1, y0)),
        (w + h, (x1, y1)),
        (2 * w + h, (x0, y1)),
    )

    found = []
    for s, corner in corners:
        d = (s - s_from) % perimeter
        if eps < d < distance - eps:
            found.append((d, corner))

    return [corner for _, corner in sorted(found)]


def _remove_spikes(points, eps):
    """Removes the zero-width spikes ``a, b, a`` of a closed polygon."""
    points = list(points)
    i = 0
    while len(points) >= 3 and i < len(points):
        j = (i + 1) % len(points)
        if _same_point(points[i - 1], points[j], eps):
            for k in sorted((i, j), reverse=True):
                points.pop(k)

            i = max(i - 2, 0)
        else:
            i += 1

    return points


def _point_in_polygon(point, polygon):
    # Even-odd rule
    x, y = point
    inside = False
    for (xa, ya), (xb, yb) in zip(polygon, polygon[1:] + polygon[:1]):
        if (ya > y) != (yb > y):
            if x < xa + (y - ya) * (xb - xa) / (yb - ya):
                inside = not inside

    return inside


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

"""
Tiles view operators.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from collections import Counter, namedtuple

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.media as fom
import fiftyone.core.stages as fosg
import fiftyone.core.tiles as fot
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.utils.tiles as fout
from fiftyone import ViewField as F

_DEFAULT_TILE_SIZE = 640
_PIXELS = "px"
_PERCENT = "%"

# Encodes an image's (width, height) as one number, to count distinct sizes
# in a single aggregation
_SIZE_KEY = 1_000_000

_ExportFormat = namedtuple(
    "_ExportFormat", ["label", "dataset_type", "label_types", "uses_classes"]
)

# The formats that the Export tiles form offers; tiles views export to every
# format from Python. Formats without label types export no label field
_EXPORT_FORMATS = {
    "yolov5": _ExportFormat(
        "YOLOv5 (Ultralytics)",
        fo.types.YOLOv5Dataset,
        (fol.Detections, fol.Polylines),
        True,
    ),
    "yolov4": _ExportFormat(
        "YOLOv4 (Darknet)",
        fo.types.YOLOv4Dataset,
        (fol.Detections, fol.Polylines),
        True,
    ),
    "coco": _ExportFormat(
        "COCO",
        fo.types.COCODetectionDataset,
        (fol.Detections, fol.Polylines, fol.Keypoints),
        True,
    ),
    "voc": _ExportFormat(
        "Pascal VOC", fo.types.VOCDetectionDataset, (fol.Detections,), False
    ),
    "kitti": _ExportFormat(
        "KITTI", fo.types.KITTIDetectionDataset, (fol.Detections,), False
    ),
    "cvat": _ExportFormat(
        "CVAT image",
        fo.types.CVATImageDataset,
        (fol.Detections, fol.Polylines, fol.Keypoints),
        False,
    ),
    "fiftyone_detection": _ExportFormat(
        "FiftyOne image detection",
        fo.types.FiftyOneImageDetectionDataset,
        (fol.Detections,),
        False,
    ),
    "segmentation": _ExportFormat(
        "Image segmentation directory",
        fo.types.ImageSegmentationDirectory,
        (fol.Segmentation, fol.Detections, fol.Polylines),
        False,
    ),
    "classification": _ExportFormat(
        "Image classification directory tree",
        fo.types.ImageClassificationDirectoryTree,
        (fol.Classification,),
        False,
    ),
    "fiftyone": _ExportFormat(
        "FiftyOne dataset (all fields)", fo.types.FiftyOneDataset, (), False
    ),
    "images": _ExportFormat("Images only", fo.types.ImageDirectory, (), False),
}

_DEFAULT_FORMAT = "yolov5"

# The formats that write a label file per image
_LABEL_FILES_FORMATS = (
    fo.types.YOLOv5Dataset,
    fo.types.YOLOv4Dataset,
    fo.types.KITTIDetectionDataset,
)

# The choices for negative tiles (without labels): in formats with a label
# file per image, they also decide which label files exist. Each set has its
# own form parameter, so that a choice made for one format does not carry
# over to another
_LABEL_FILES_EMPTY_TILES = (
    "label_files_empty_tiles",
    (
        (
            "keep",
            "Keep, with empty labels",
            "Exported, with empty label files",
        ),
        (
            "keep_without_labels",
            "Keep, without labels",
            "Exported, without label files",
        ),
        ("skip", "Skip", "Not exported"),
    ),
)
_EMPTY_TILES = (
    "empty_tiles",
    (("keep", "Keep", "Exported"), ("skip", "Skip", "Not exported")),
)

_YOLO_SPLITS = ("train", "val", "test")

# Where the splits of the images come from
_SPLIT_BY = (
    ("none", "None"),
    ("tags", "Sample tags"),
    ("field", "Field"),
    ("random", "Random"),
)

# The sample tags that the form proposes as splits, when the images have them
_SPLIT_TAGS = ("train", "val", "validation", "test")

# The splits of a random split, and their default fractions
_RANDOM_SPLITS = (("train", 0.8), ("val", 0.2), ("test", 0.0))
_DEFAULT_SEED = 51

_MAX_CLASSES_SHOWN = 8


class PreviewTiling(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="preview_tiling",
            label="Preview tiling",
            description=(
                "Previews the images of the current view split into "
                "fixed-size tiles"
            ),
            dynamic=True,
            icon="grid_view",
            risk_level=types.RiskLevel.LOW,
        )

    def resolve_placement(self, ctx):
        if not _is_image_view(ctx):
            return None

        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Preview tiling", icon="grid_view", prompt=True
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        if not _is_image_view(ctx):
            prop = inputs.str(
                "unsupported",
                label="Tiling previews are only available for images",
                view=types.Warning(),
            )
            prop.invalid = True
            return types.Property(
                inputs, view=types.View(label="Preview tiling")
            )

        defaults = _get_current_tiling(ctx)

        inputs.int(
            "tile_width",
            required=True,
            min=1,
            default=defaults["tile_width"],
            label="Tile width (px)",
        )
        inputs.int(
            "tile_height",
            required=True,
            min=1,
            default=defaults["tile_height"],
            label="Tile height (px)",
        )
        inputs.float(
            "overlap",
            min=0,
            default=defaults["overlap"],
            label="Overlap",
            description=(
                "Overlap between adjacent tiles. The last tile of each row "
                "and column is shifted back to end at the image edge, so its "
                "overlap can be larger"
            ),
        )
        unit_choices = types.RadioGroup(orientation="horizontal")
        unit_choices.add_choice(_PIXELS, label="pixels")
        unit_choices.add_choice(_PERCENT, label="% of the tile size")
        inputs.enum(
            "overlap_unit",
            unit_choices.values(),
            default=defaults["overlap_unit"],
            label="Overlap unit",
            view=unit_choices,
        )
        inputs.float(
            "min_label_coverage",
            min=0,
            max=100,
            default=defaults["min_label_coverage"],
            label="Min label coverage (%)",
            description=(
                "Tiles only keep labels with at least this share of their "
                "area inside the tile. 0 keeps every label that touches the "
                "tile"
            ),
        )

        base_view = _get_base_view(ctx)

        num_missing = _count_missing_metadata(base_view)
        if num_missing:
            inputs.bool(
                "compute_metadata",
                default=False,
                label=(
                    "Compute metadata for the %d image(s) that do not have "
                    "it yet" % num_missing
                ),
                description=(
                    "Tiling needs each image's width and height. This "
                    "writes them to those samples' metadata field"
                ),
            )
            if not ctx.params.get("compute_metadata", False):
                prop = inputs.str(
                    "missing_metadata",
                    label=(
                        "%d image(s) have no metadata, so they cannot be "
                        "tiled yet" % num_missing
                    ),
                    view=types.Warning(),
                )
                prop.invalid = True

        try:
            tile_size, overlap, _ = _parse_params(ctx.params)
        except ValueError as e:
            prop = inputs.str(
                "invalid_params", label=str(e), view=types.Error()
            )
            prop.invalid = True
        else:
            summary = _summarize_tiling(base_view, tile_size, overlap)
            if summary:
                inputs.md(summary, name="summary")

        return types.Property(inputs, view=types.View(label="Preview tiling"))

    def execute(self, ctx):
        tile_size, overlap, min_label_coverage = _parse_params(ctx.params)

        base_view = _get_base_view(ctx)
        if ctx.params.get("compute_metadata", False):
            base_view.compute_metadata()

        view = base_view.to_tiles(
            tile_size, overlap=overlap, min_label_coverage=min_label_coverage
        )
        for stage in _get_following_stages(ctx):
            view = view.add_stage(stage)

        ctx.ops.set_view(view)


def _is_image_view(ctx):
    return ctx.dataset is not None and ctx.view.media_type == fom.IMAGE


def _get_tiles_stage(ctx):
    for stage in ctx.view.view()._all_stages:
        if isinstance(stage, fosg.ToTiles):
            return stage

    return None


def _get_base_view(ctx):
    """The current view up to (excluding) any tiling it already has."""
    view = ctx.dataset.view()
    for stage in ctx.view.view()._all_stages:
        if isinstance(stage, fosg.ToTiles):
            break

        view = view.add_stage(stage)

    return view


def _get_following_stages(ctx):
    """The stages that the current view applies after its tiling."""
    stages = []
    found = False
    for stage in ctx.view.view()._all_stages:
        if found:
            stages.append(stage)

        if isinstance(stage, fosg.ToTiles):
            found = True

    return stages


def _get_current_tiling(ctx):
    """Form defaults: the current view's tiling, if any."""
    stage = _get_tiles_stage(ctx)
    if stage is None:
        return {
            "tile_width": _DEFAULT_TILE_SIZE,
            "tile_height": _DEFAULT_TILE_SIZE,
            "overlap": 0,
            "overlap_unit": _PIXELS,
            "min_label_coverage": 0,
        }

    tile_width, tile_height = stage.tile_size
    overlap = stage.overlap
    if 0 < overlap < 1:
        overlap, overlap_unit = round(100 * overlap, 6), _PERCENT
    else:
        overlap_unit = _PIXELS

    return {
        "tile_width": tile_width,
        "tile_height": tile_height,
        "overlap": overlap,
        "overlap_unit": overlap_unit,
        "min_label_coverage": round(100 * stage.min_label_coverage, 6),
    }


def _parse_params(params):
    """Converts the form's values to ``to_tiles()`` arguments.

    Raises:
        ValueError: if the values do not describe a valid tiling
    """
    tile_width = params.get("tile_width", None)
    tile_height = params.get("tile_height", None)
    if not tile_width or not tile_height:
        raise ValueError("Enter a tile width and height")

    tile_size = (int(tile_width), int(tile_height))

    overlap = params.get("overlap", None) or 0
    if params.get("overlap_unit", _PIXELS) == _PERCENT:
        if not 0 <= overlap < 100:
            raise ValueError("The overlap must be less than 100%")

        overlap /= 100
    elif 0 < overlap < 1:
        # to_tiles() reads values below 1 as fractions, not pixels
        raise ValueError("A pixel overlap must be a whole number of pixels")
    else:
        overlap = int(round(overlap))

    min_label_coverage = (params.get("min_label_coverage", None) or 0) / 100
    if not 0 <= min_label_coverage <= 1:
        raise ValueError("The min label coverage must be between 0 and 100%")

    # Raises if the overlap is too large for the tile size
    fot._parse_tiling(tile_size, overlap)

    return tile_size, overlap, min_label_coverage


def _count_missing_metadata(view):
    has_size = (F("metadata.width") > 0) & (F("metadata.height") > 0)
    return view.match(~has_size).count()


def _with_size(view):
    """The images of the view whose size is known."""
    has_size = (F("metadata.width") > 0) & (F("metadata.height") > 0)
    return view.match(has_size)


def _count_image_sizes(view):
    """Counts the images of each ``(width, height)`` in the view."""
    key = F("metadata.width") * _SIZE_KEY + F("metadata.height")
    counts = _with_size(view).count_values(key)

    sizes = Counter()
    for key, count in counts.items():
        key = int(key)
        sizes[(key // _SIZE_KEY, key % _SIZE_KEY)] += count

    return sizes


def _summarize_tiling(view, tile_size, overlap):
    """A markdown summary of how the view's images would be tiled."""
    sizes = _count_image_sizes(view)
    num_images = sum(sizes.values())
    if not num_images:
        return None

    tile_w, tile_h = tile_size
    num_tiles = 0
    grids = Counter()
    num_clamped = 0
    for (width, height), count in sizes.items():
        tiles = fot.compute_tiles(width, height, tile_size, overlap=overlap)
        rows = tiles[-1]["row"] + 1
        cols = tiles[-1]["col"] + 1
        num_tiles += count * len(tiles)
        grids[(cols, rows)] += count
        if width < tile_w or height < tile_h:
            num_clamped += count

    lines = [
        "**%s image(s) → %s tiles** (%.1f per image on average)"
        % (
            "{:,}".format(num_images),
            "{:,}".format(num_tiles),
            num_tiles / num_images,
        )
    ]

    if len(grids) == 1:
        (cols, rows), _ = grids.most_common(1)[0]
        lines.append("Every image is cut into %d × %d tiles" % (cols, rows))
    else:
        (cols, rows), count = grids.most_common(1)[0]
        lines.append(
            "Most common grid: %d × %d tiles (%s image(s)), out of %d "
            "different grids" % (cols, rows, "{:,}".format(count), len(grids))
        )

    _, _, overlap_x, overlap_y = fot._parse_tiling(tile_size, overlap)
    if overlap_x or overlap_y:
        lines.append("Overlap: %d × %d px" % (overlap_x, overlap_y))

    if num_clamped:
        lines.append(
            "%s image(s) are smaller than a tile in at least one dimension; "
            "their tiles are clamped to the image size"
            % "{:,}".format(num_clamped)
        )

    lines.extend(_summarize_uncontained_labels(view, tile_size, overlap))

    return "\n\n".join(lines)


def _summarize_uncontained_labels(view, tile_size, overlap):
    """Markdown lines on the objects that no tile would fully contain, which
    shows whether the tiles or their overlap are too small for the objects.
    """
    counts = fot.count_uncontained_labels(
        _with_size(view), tile_size, overlap=overlap
    )

    items = []
    for field, c in counts.items():
        if not c["count"]:
            continue

        if not c["uncontained"]:
            items.append(
                "- `%s`: none of %s" % (field, "{:,}".format(c["count"]))
            )
            continue

        too_large = c["too_large"]
        crossing = c["uncontained"] - too_large
        reasons = []
        if too_large:
            reasons.append("%s larger than a tile" % "{:,}".format(too_large))

        if crossing:
            reasons.append(
                "%s crossing tile borders" % "{:,}".format(crossing)
            )

        items.append(
            "- `%s`: %s of %s (%s) — %s"
            % (
                field,
                "{:,}".format(c["uncontained"]),
                "{:,}".format(c["count"]),
                _format_percent(c["uncontained"], c["count"]),
                ", ".join(reasons),
            )
        )

    if not items:
        return []

    return ["Objects that no tile fully contains:\n\n" + "\n".join(items)]


def _format_percent(part, total):
    percent = 100 * part / total
    if 0 < percent < 0.1:
        return "<0.1%"

    if 99.9 < percent < 100:
        return ">99.9%"

    return "%.1f%%" % percent


class ExportTiles(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="export_tiles",
            label="Export tiles",
            description=(
                "Exports the tiles of the current tiles view as images, "
                "with their labels clipped to the tiles"
            ),
            dynamic=True,
            icon="file_download",
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=False,
        )

    def resolve_placement(self, ctx):
        if not _is_tiles_view(ctx):
            return None

        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Export tiles", icon="file_download", prompt=True
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        view = types.View(label="Export tiles")

        if not _is_tiles_view(ctx):
            prop = inputs.str(
                "not_tiles",
                label=(
                    "Export tiles works on tiles views: use Preview tiling "
                    "first"
                ),
                view=types.Warning(),
            )
            prop.invalid = True
            return types.Property(inputs, view=view)

        format_choices = types.Dropdown()
        for key, fmt in _EXPORT_FORMATS.items():
            format_choices.add_choice(key, label=fmt.label)

        inputs.enum(
            "format",
            format_choices.values(),
            required=True,
            default=_DEFAULT_FORMAT,
            label="Format",
            view=format_choices,
        )
        fmt = _EXPORT_FORMATS[ctx.params.get("format") or _DEFAULT_FORMAT]

        label_field = None
        if fmt.label_types:
            fields = _get_export_fields(ctx, fmt)
            if not fields:
                prop = inputs.str(
                    "no_fields",
                    label=(
                        "The tiles have no %s field to export in this format"
                        % " or ".join(t.__name__ for t in fmt.label_types)
                    ),
                    view=types.Warning(),
                )
                prop.invalid = True
                return types.Property(inputs, view=view)

            field_choices = types.Dropdown()
            for field in fields:
                field_choices.add_choice(field, label=field)

            inputs.enum(
                "label_field",
                field_choices.values(),
                required=True,
                default=fields[0],
                label="Label field",
                view=field_choices,
            )
            label_field = ctx.params.get("label_field")
            if label_field not in fields:
                label_field = fields[0]

        inputs.file(
            "export_dir",
            required=True,
            label="Export directory",
            description="The directory in which to write the tiles",
            view=types.FileExplorerView(
                choose_dir=True, button_label="Choose a directory..."
            ),
        )

        if label_field is not None:
            param, choices = _get_empty_tiles_choices(fmt)
            empty_choices = types.RadioGroup()
            for value, label, description in choices:
                empty_choices.add_choice(
                    value, label=label, description=description
                )

            inputs.enum(
                param,
                empty_choices.values(),
                default="keep",
                label="Negative Tiles (i.e., without labels)",
                description=(
                    "Tiles without labels in %s, because none of their "
                    "image's labels touch them or clipping removed them all"
                    % label_field
                ),
                view=empty_choices,
            )

        if label_field is not None and _is_polylines_field(ctx, label_field):
            inputs.bool(
                "join_polygon_parts",
                default=False,
                label="Join polygon parts",
                description=(
                    "Keep a polygon that a tile cuts into several parts as "
                    "one polygon, its parts joined along the tile's border, "
                    "rather than one polygon per part"
                ),
            )

        split_by, split_params = _add_split_inputs(ctx, inputs, fmt)

        if split_by == "none" and fmt.dataset_type is fo.types.YOLOv5Dataset:
            split_choices = types.Dropdown()
            for split in _YOLO_SPLITS:
                split_choices.add_choice(split, label=split)

            inputs.enum(
                "split",
                split_choices.values(),
                default="train",
                label="Split",
                description=(
                    "The YOLOv5 split in which to export all the tiles"
                ),
                view=split_choices,
            )

        inputs.bool(
            "overwrite",
            default=False,
            label="Delete the export directory first",
            description=(
                "By default, the export is merged into the directory, as "
                "when exporting several splits"
            ),
        )

        summary = _summarize_export(ctx, fmt, label_field)
        if split_by != "none":
            split_summary = _summarize_splits(ctx, inputs, split_params)
            if split_summary is None:
                return types.Property(inputs, view=view)

            summary += "\n\n" + split_summary

        inputs.md(summary, name="summary")

        return types.Property(inputs, view=view)

    def execute(self, ctx):
        fmt = _EXPORT_FORMATS[ctx.params.get("format") or _DEFAULT_FORMAT]
        export_dir = _get_export_dir(ctx.params)
        if not export_dir:
            raise ValueError("Choose an export directory")

        kwargs = {
            "export_dir": export_dir,
            "dataset_type": fmt.dataset_type,
            "overwrite": bool(ctx.params.get("overwrite", False)),
        }

        if fmt.label_types:
            label_field = ctx.params.get("label_field")
            if label_field not in _get_export_fields(ctx, fmt):
                raise ValueError("Choose a label field to export")

            kwargs["label_field"] = label_field
            param, choices = _get_empty_tiles_choices(fmt)
            empty_tiles = ctx.params.get(param) or "keep"
            if empty_tiles not in [c[0] for c in choices]:
                raise ValueError("Invalid choice %r" % empty_tiles)

            kwargs["empty_tiles"] = empty_tiles
            kwargs["join_polygon_parts"] = bool(
                ctx.params.get("join_polygon_parts", False)
            )

            if fmt.uses_classes:
                classes = _get_classes(ctx, label_field)
                if classes:
                    kwargs["classes"] = classes

        split_kwargs = _get_split_kwargs(ctx.params)
        if split_kwargs is None:
            if fmt.dataset_type is fo.types.YOLOv5Dataset:
                kwargs["split"] = ctx.params.get("split") or "train"

            num_tiles = ctx.view.export(
                progress=_ExportProgress(ctx), **kwargs
            )
            return {"num_tiles": num_tiles, "export_dir": export_dir}

        split_ids = fout.get_splits(ctx.view, **split_kwargs)
        num_writes = _count_split_writes(ctx, fmt, split_ids)
        counts = ctx.view.export(
            progress=_ExportProgress(ctx, num_writes=num_writes),
            **split_kwargs,
            **kwargs,
        )

        if ctx.params.get("split_by") == "random" and ctx.params.get(
            "tag_splits", False
        ):
            _tag_splits(ctx, split_ids)

        return {
            "num_tiles": sum(counts.values()),
            "splits": ", ".join(
                "%s: %d" % (name, count) for name, count in counts.items()
            ),
            "export_dir": export_dir,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("num_tiles", label="Tiles exported")
        if ctx.results.get("splits"):
            outputs.str("splits", label="Tiles per split")

        outputs.str("export_dir", label="Export directory")
        return types.Property(outputs, view=types.View(label="Export tiles"))


class _ExportProgress(object):
    """Reports the progress of a tiles export, which first crops the tiles
    and then writes them, in one or more exports (one per split). Cropping
    and writing each take half of the progress bar.
    """

    def __init__(self, ctx, num_writes=1):
        self._ctx = ctx
        self._num_writes = max(num_writes, 1)
        self._bars = []

    def __call__(self, pb):
        if not self._bars or self._bars[-1] is not pb:
            self._bars.append(pb)

        progress = pb.progress or 0
        if len(self._bars) == 1:
            self._ctx.set_progress(
                progress=progress / 2, label="Cropping the tiles"
            )
            return

        write = min(len(self._bars) - 1, self._num_writes)
        label = "Writing the export"
        if self._num_writes > 1:
            label += " (%d of %d)" % (write, self._num_writes)

        progress = 0.5 + (write - 1 + progress) / (2 * self._num_writes)
        self._ctx.set_progress(progress=progress, label=label)


def _add_split_inputs(ctx, inputs, fmt):
    """Adds the inputs that split the images, and returns where their splits
    come from and the form's parameters, with the defaults of the split
    inputs that have no value yet, or None if the splits are invalid.
    """
    params = dict(ctx.params)

    split_choices = types.RadioGroup()
    for value, label in _SPLIT_BY:
        split_choices.add_choice(value, label=label)

    inputs.enum(
        "split_by",
        split_choices.values(),
        default="none",
        label="Splits",
        description=(
            "Where each image's split comes from; all the tiles of an image "
            "go to its split. %s" % _describe_split_layout(fmt)
        ),
        view=split_choices,
    )

    split_by = params.get("split_by") or "none"

    if split_by == "tags":
        tags = _get_source_images(ctx).distinct("tags")
        if not tags:
            prop = inputs.str(
                "no_tags",
                label="The images have no tags",
                view=types.Warning(),
            )
            prop.invalid = True
            return split_by, None

        default = [t for t in _SPLIT_TAGS if t in tags]
        if params.get("split_tags") is None:
            params["split_tags"] = default

        inputs.list(
            "split_tags",
            types.String(),
            default=default,
            required=True,
            label="Split tags",
            description="The tags of the images that are splits",
            view=types.AutocompleteView(
                multiple=True,
                choices=[types.Choice(t, label=t) for t in tags],
                allow_user_input=False,
                allow_duplicates=False,
            ),
        )
    elif split_by == "field":
        fields = _get_split_fields(ctx)
        if not fields:
            prop = inputs.str(
                "no_split_fields",
                label="The images have no string field",
                view=types.Warning(),
            )
            prop.invalid = True
            return split_by, None

        if params.get("split_field") not in fields:
            params["split_field"] = fields[0]

        field_choices = types.Dropdown()
        for field in fields:
            field_choices.add_choice(field, label=field)

        inputs.enum(
            "split_field",
            field_choices.values(),
            default=fields[0],
            required=True,
            label="Split field",
            description=(
                "A string field of the images, whose values are their splits"
            ),
            view=field_choices,
        )
    elif split_by == "random":
        fracs = types.Object()
        for name, frac in _RANDOM_SPLITS:
            fracs.float(name, default=frac, min=0, label=name)

        inputs.define_property(
            "split_fracs",
            fracs,
            label="Split fractions",
            description="Normalized to sum to 1; splits of 0 are left out",
            view=types.HStackView(),
        )
        inputs.int(
            "split_seed",
            default=_DEFAULT_SEED,
            label="Random seed",
            description="The same seed and images give the same split",
        )
        inputs.bool(
            "tag_splits",
            default=False,
            label="Tag the images with their split",
            description=(
                "After the export, replaces the train, val, and test tags of "
                "the images with their split, to reuse it as sample tags"
            ),
        )

    return split_by, params


def _describe_split_layout(fmt):
    if fmt.dataset_type is fo.types.YOLOv5Dataset:
        return (
            "Splits go to images/<split> and labels/<split>, listed in one "
            "dataset.yaml"
        )

    if fmt.dataset_type is fo.types.FiftyOneDataset:
        return "Tiles are tagged with their split"

    return "Each split goes to its own <split> directory"


def _get_split_kwargs(params):
    """The split arguments of :meth:`fiftyone.utils.tiles.get_splits` that
    the form's parameters define, or None if the images are not split.
    """
    split_by = params.get("split_by") or "none"
    if split_by == "none":
        return None

    if split_by == "tags":
        split_tags = params.get("split_tags") or []
        if not split_tags:
            raise ValueError("Choose the tags that are splits")

        return {"splits": list(split_tags), "split_field": "tags"}

    if split_by == "field":
        split_field = params.get("split_field")
        if not split_field:
            raise ValueError("Choose the field that holds the splits")

        return {"split_field": split_field}

    if split_by == "random":
        fracs = params.get("split_fracs") or {}
        split_fracs = {}
        for name, default in _RANDOM_SPLITS:
            frac = fracs.get(name)
            frac = default if frac is None else float(frac)
            if frac < 0:
                raise ValueError("Split fractions cannot be negative")

            if frac > 0:
                split_fracs[name] = frac

        if not split_fracs:
            raise ValueError("Give at least one split a fraction above 0")

        seed = params.get("split_seed")
        return {
            "splits": split_fracs,
            "seed": _DEFAULT_SEED if seed is None else int(seed),
        }

    raise ValueError("Invalid choice %r" % split_by)


def _summarize_splits(ctx, inputs, params):
    """A markdown summary of the splits, or None if they are invalid, in
    which case the inputs have an invalid warning.
    """
    if params is None:
        return None

    def _invalid(message):
        prop = inputs.str(
            "invalid_splits", label=message, view=types.Warning()
        )
        prop.invalid = True

    try:
        split_kwargs = _get_split_kwargs(params)
        split_ids = fout.get_splits(ctx.view, **split_kwargs)
    except ValueError as e:
        _invalid(str(e))
        return None

    image_splits = Counter(_id for ids in split_ids.values() for _id in ids)
    num_shared = sum(1 for c in image_splits.values() if c > 1)
    if num_shared:
        _invalid(
            "%s image(s) are in more than one split"
            % "{:,}".format(num_shared)
        )
        return None

    if not split_ids:
        _invalid("The images have no splits")
        return None

    num_tiles = ctx.view.count_values("sample_id")
    lines = []
    for name, ids in split_ids.items():
        lines.append(
            "- `%s`: %s tile(s) of %s image(s)"
            % (
                name,
                "{:,}".format(sum(num_tiles.get(_id, 0) for _id in ids)),
                "{:,}".format(len(ids)),
            )
        )

    num_unassigned = len(num_tiles) - len(image_splits)
    if num_unassigned:
        lines.append(
            "\n%s image(s) are in no split: their tiles are not exported"
            % "{:,}".format(num_unassigned)
        )

    return "\n".join(lines)


def _count_split_writes(ctx, fmt, split_ids):
    """The number of exports that write the splits."""
    if fmt.dataset_type is fo.types.FiftyOneDataset:
        return 1

    num_tiles = ctx.view.count_values("sample_id")
    return sum(
        1
        for ids in split_ids.values()
        if any(num_tiles.get(_id) for _id in ids)
    )


def _tag_splits(ctx, split_ids):
    """Tags the source images with their split, replacing their tags of the
    random splits.
    """
    dataset = ctx.view._root_dataset
    names = [name for name, _ in _RANDOM_SPLITS]
    ids = [_id for ids in split_ids.values() for _id in ids]
    dataset.select(ids).untag_samples(names)
    for name, ids in split_ids.items():
        if ids:
            dataset.select(ids).tag_samples(name)


def _get_source_images(ctx):
    """The source images of the tiles view."""
    return ctx.view._source_collection.select(ctx.view.distinct("sample_id"))


def _get_split_fields(ctx):
    """The top-level string fields of the source images, which may hold
    their splits.
    """
    schema = ctx.view._source_collection.get_field_schema(ftype=fo.StringField)
    return [f for f in schema if f != "filepath"]


def _get_empty_tiles_choices(fmt):
    """The form parameter and choices for negative tiles in the format."""
    if fmt.dataset_type in _LABEL_FILES_FORMATS:
        return _LABEL_FILES_EMPTY_TILES

    return _EMPTY_TILES


def _is_tiles_view(ctx):
    return ctx.dataset is not None and isinstance(ctx.view, fot.TilesView)


def _get_export_fields(ctx, fmt):
    """The top-level fields of the tiles view that the format can export."""
    schema = ctx.view.get_field_schema(embedded_doc_type=fmt.label_types)
    return [f for f in schema if f != fot.TILE_FIELD]


def _is_polylines_field(ctx, field):
    doc_type = getattr(ctx.view.get_field(field), "document_type", None)
    return doc_type is not None and issubclass(doc_type, fol.Polylines)


def _get_export_dir(params):
    export_dir = params.get("export_dir")
    if isinstance(export_dir, dict):
        export_dir = export_dir.get("absolute_path")

    return export_dir or None


def _get_classes(ctx, label_field):
    """The classes of a label field of the whole source dataset, so that
    class indices do not change across the splits or tilings exported.
    """
    dataset = ctx.dataset
    classes = dataset.classes.get(label_field) or dataset.default_classes
    if classes:
        return list(classes)

    doc_type = getattr(dataset.get_field(label_field), "document_type", None)
    list_field = getattr(doc_type, "_LABEL_LIST_FIELD", None)
    if list_field is None:
        return None

    return dataset.distinct("%s.%s.label" % (label_field, list_field))


def _summarize_export(ctx, fmt, label_field):
    """A markdown summary of what the export writes."""
    num_tiles = len(ctx.view)
    lines = [
        "**%s tile(s)** of %s image(s)"
        % (
            "{:,}".format(num_tiles),
            "{:,}".format(len(ctx.view.distinct("sample_id"))),
        )
    ]

    if label_field is not None:
        has_labels = _has_labels_expr(ctx.view, label_field)
        num_empty = num_tiles - ctx.view.match(has_labels).count()
        lines.append(
            "%s tile(s) have no labels in `%s`; clipping may leave a few "
            "more without labels" % ("{:,}".format(num_empty), label_field)
        )

    if fmt.uses_classes and label_field is not None:
        classes = _get_classes(ctx, label_field)
        if classes:
            shown = ", ".join(
                "%d `%s`" % (i, c)
                for i, c in enumerate(classes[:_MAX_CLASSES_SHOWN])
            )
            if len(classes) > _MAX_CLASSES_SHOWN:
                shown += ", and %d more" % (len(classes) - _MAX_CLASSES_SHOWN)

            lines.append(
                "%d classes, from the whole dataset so that their indices do "
                "not change across exports: %s" % (len(classes), shown)
            )

    return "\n\n".join(lines)


def _has_labels_expr(view, label_field):
    doc_type = getattr(view.get_field(label_field), "document_type", None)
    list_field = getattr(doc_type, "_LABEL_LIST_FIELD", None)
    if list_field is not None:
        return F("%s.%s" % (label_field, list_field)).length() > 0

    return F(label_field) != None


def register(p):
    p.register(PreviewTiling)
    p.register(ExportTiles)

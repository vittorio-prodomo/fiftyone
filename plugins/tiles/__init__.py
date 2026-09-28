"""
Tiles view operators.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from collections import Counter

import fiftyone.core.media as fom
import fiftyone.core.stages as fosg
import fiftyone.core.tiles as fot
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone import ViewField as F

_DEFAULT_TILE_SIZE = 640
_PIXELS = "px"
_PERCENT = "%"

# Encodes an image's (width, height) as one number, to count distinct sizes
# in a single aggregation
_SIZE_KEY = 1_000_000


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


def _count_image_sizes(view):
    """Counts the images of each ``(width, height)`` in the view."""
    has_size = (F("metadata.width") > 0) & (F("metadata.height") > 0)
    key = F("metadata.width") * _SIZE_KEY + F("metadata.height")
    counts = view.match(has_size).count_values(key)

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

    return "\n\n".join(lines)


def register(p):
    p.register(PreviewTiling)

"""
Tile generation utilities.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import fiftyone.core.labels as fol


def compute_tile_detections(
    img_w, img_h, tile_w, tile_h, overlap=0, min_coverage=0.0
):
    """Computes a grid of tile bounding boxes for an image.

    Args:
        img_w: image width in pixels
        img_h: image height in pixels
        tile_w: tile width in pixels
        tile_h: tile height in pixels
        overlap: overlap between adjacent tiles. Values >= 1 are pixels,
            values < 1 are fractions of tile size
        min_coverage: minimum fraction of tile area that must be within the
            image for edge tiles (0.0 to keep all tiles)

    Returns:
        a list of :class:`fiftyone.core.labels.Detection` instances
    """
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError(
            "Tile dimensions must be positive, got (%d x %d)"
            % (tile_w, tile_h)
        )

    if img_w <= 0 or img_h <= 0:
        raise ValueError(
            "Image dimensions must be positive, got (%d x %d)" % (img_w, img_h)
        )

    if overlap < 0:
        raise ValueError("Overlap must be >= 0, got %s" % overlap)

    overlap_x = overlap * tile_w if overlap < 1 else overlap
    overlap_y = overlap * tile_h if overlap < 1 else overlap

    stride_x = tile_w - overlap_x
    stride_y = tile_h - overlap_y

    if stride_x <= 0 or stride_y <= 0:
        raise ValueError(
            "Overlap (%s) must be less than tile size (%d x %d)"
            % (overlap, tile_w, tile_h)
        )

    detections = []
    row = 0
    while row * stride_y < img_h:
        y_px = row * stride_y
        col = 0
        while col * stride_x < img_w:
            x_px = col * stride_x
            actual_w = min(tile_w, img_w - x_px)
            actual_h = min(tile_h, img_h - y_px)

            coverage = (actual_w * actual_h) / (tile_w * tile_h)
            if coverage >= min_coverage:
                detections.append(
                    fol.Detection(
                        label="tile_%d_%d" % (row, col),
                        bounding_box=[
                            x_px / img_w,
                            y_px / img_h,
                            actual_w / img_w,
                            actual_h / img_h,
                        ],
                    )
                )

            col += 1

        row += 1

    return detections

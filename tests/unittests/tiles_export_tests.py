"""
FiftyOne tiles materialization and export unit tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import json
import math
import os
import random
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageOps

import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.core.tiles as fot
import fiftyone.utils.tiles as fout
import fiftyone.utils.yolo as fouy

from decorators import drop_datasets

nan = float("nan")


def _coords_image(width, height):
    """An RGB image whose pixel at (x, y) is (x % 256, y % 256, x // 256 +
    16 * (y // 256)), so that any crop reveals where it was taken.
    """
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    return np.stack(
        [xs % 256, ys % 256, xs // 256 + 16 * (ys // 256)], axis=-1
    ).astype(np.uint8)


def _write_png(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


def _read(path):
    with Image.open(path) as img:
        return np.asarray(img)


def _area(points):
    area = 0.0
    for (xa, ya), (xb, yb) in zip(points, points[1:] + points[:1]):
        area += xa * yb - xb * ya

    return abs(area) / 2


def _inside(point, polygon):
    # Even-odd rule
    x, y = point
    inside = False
    for (xa, ya), (xb, yb) in zip(polygon, polygon[1:] + polygon[:1]):
        if (ya > y) != (yb > y):
            if x < xa + (y - ya) * (xb - xa) / (yb - ya):
                inside = not inside

    return inside


def _assert_box(det, expected):
    np.testing.assert_allclose(det.bounding_box, expected, atol=1e-9)


class ClipLabelTests(unittest.TestCase):
    """A 1000x500 image, clipped to the tile at x=400, y=100 of 400x300."""

    TILE = (400, 100, 400, 300)
    SIZE = (1000, 500)

    def _clip(self, label, **kwargs):
        return fout.clip_label(label, self.TILE, self.SIZE, **kwargs)

    def test_detection(self):
        # inside: 500-600 x 200-300
        det = fo.Detection(label="a", bounding_box=[0.5, 0.4, 0.1, 0.2])
        clipped = self._clip(det)
        _assert_box(clipped, [0.25, 1 / 3, 0.25, 1 / 3])
        self.assertEqual(clipped.label, "a")
        self.assertNotEqual(clipped.id, det.id)

        # partly outside: 300-500 x 0-200 -> 400-500 x 100-200
        det = fo.Detection(bounding_box=[0.3, 0, 0.2, 0.4], confidence=0.9)
        clipped = self._clip(det)
        _assert_box(clipped, [0, 0, 0.25, 1 / 3])
        self.assertEqual(clipped.confidence, 0.9)

        # outside, and touching the tile's edge
        for box in ([0, 0, 0.1, 0.1], [0.3, 0.4, 0.1, 0.1]):
            self.assertIsNone(self._clip(fo.Detection(bounding_box=box)))

        # no box
        self.assertIsNone(self._clip(fo.Detection(label="a")))

    def test_detection_edges_despite_float_rounding(self):
        # on a 1333px wide image, a box from 1px to 54px ends after 54px
        frame_size = (1333, 500)
        tile = (54, 0, 300, 500)
        self.assertGreater((1 / 1333) * 1333 + (53 / 1333) * 1333, 54)

        # so it touches the tile, and is not inside it
        det = fo.Detection(bounding_box=[1 / 1333, 0.1, 53 / 1333, 0.1])
        self.assertIsNone(fout.clip_label(det, tile, frame_size))

        # and the tile to its left contains it entirely
        clipped = fout.clip_label(det, (0, 0, 54, 500), frame_size)
        _assert_box(clipped, [1 / 54, 0.1, 53 / 54, 0.1])

    def test_detection_masks(self):
        # 300-500 x 200-300, with a 2x lower resolution mask whose columns
        # count up
        mask = np.tile(np.arange(100, dtype=np.uint8), (50, 1))
        det = fo.Detection(bounding_box=[0.3, 0.4, 0.2, 0.2], mask=mask)

        # only 400-500 is inside: the right half of the mask
        clipped = self._clip(det)
        self.assertEqual(clipped.mask.shape, (50, 50))
        self.assertEqual(clipped.mask[0, 0], 50)
        self.assertEqual(det.mask.shape, (50, 100))

        # masks on disk are cropped as stored, into the mask directory
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "mask.png")
            Image.fromarray(mask).save(path)
            det = fo.Detection(
                bounding_box=[0.3, 0.4, 0.2, 0.2], mask_path=path
            )

            clipped = self._clip(det, mask_dir=os.path.join(tmp, "out"))
            self.assertIsNone(clipped.mask)
            self.assertTrue(clipped.mask_path.startswith(tmp + "/out/"))
            self.assertEqual(_read(clipped.mask_path)[0, 0], 50)
            self.assertEqual(_read(path).shape, (50, 100))

            # or into memory
            clipped = self._clip(det)
            self.assertIsNone(clipped.mask_path)
            self.assertEqual(clipped.mask.shape, (50, 50))
        finally:
            shutil.rmtree(tmp)

    def test_truncated(self):
        # 300-500 x 200-300: half of it is inside the tile (400-800)
        def clip(mode=None, **attributes):
            det = fo.Detection(bounding_box=[0.3, 0.4, 0.2, 0.2], **attributes)
            clipped = self._clip(det, truncated=mode)
            return clipped.get_attribute_value("truncated", None)

        # existing flags and fractions are updated, keeping their type
        self.assertIs(clip(truncated=False), True)
        self.assertEqual(clip(truncated=0), 1)
        self.assertIsInstance(clip(truncated=0), int)
        self.assertAlmostEqual(clip(truncated=0.2), 1 - 0.8 * 0.5)

        # missing ones are only added on request
        self.assertIsNone(clip())
        self.assertEqual(clip("flag"), 1)
        self.assertAlmostEqual(clip("fraction"), 0.5)

        # boxes inside the tile are left as they are
        det = fo.Detection(bounding_box=[0.5, 0.4, 0.1, 0.1], truncated=0.3)
        self.assertEqual(
            self._clip(det, truncated="fraction")["truncated"], 0.3
        )
        det = fo.Detection(bounding_box=[0.5, 0.4, 0.1, 0.1])
        clipped = self._clip(det, truncated="flag")
        self.assertIsNone(clipped.get_attribute_value("truncated", None))

        with self.assertRaises(ValueError):
            clip("yes")

    def test_random_detections_match_the_intersection(self):
        rng = random.Random(0)
        tile_x, tile_y, tile_w, tile_h = self.TILE
        for _ in range(500):
            x, y = rng.randint(0, 999), rng.randint(0, 499)
            w, h = rng.randint(1, 1000 - x), rng.randint(1, 500 - y)
            det = fo.Detection(
                bounding_box=[x / 1000, y / 500, w / 1000, h / 500]
            )

            x1, y1 = max(x, tile_x), max(y, tile_y)
            x2 = min(x + w, tile_x + tile_w)
            y2 = min(y + h, tile_y + tile_h)

            clipped = self._clip(det)
            if x2 <= x1 or y2 <= y1:
                self.assertIsNone(clipped, msg=(x, y, w, h))
                continue

            expected = [
                (x1 - tile_x) / tile_w,
                (y1 - tile_y) / tile_h,
                (x2 - x1) / tile_w,
                (y2 - y1) / tile_h,
            ]
            np.testing.assert_allclose(
                clipped.bounding_box, expected, atol=1e-12
            )

    def test_filled_polygons(self):
        # a square half inside: 300-500 x 200-300
        square = [(0.3, 0.4), (0.5, 0.4), (0.5, 0.6), (0.3, 0.6)]
        poly = fo.Polyline(points=[square], closed=True, filled=True)
        clipped = self._clip(poly)
        self.assertEqual(len(clipped.points), 1)
        self.assertAlmostEqual(_area(clipped.points[0]), 0.25 * (1 / 3))
        self.assertTrue(clipped.closed and clipped.filled)

        # a U whose arms are 100px wide, which the tile (y < 400px) cuts into
        # its two arms
        u_shape = [
            (0.45, 0.1),
            (0.55, 0.1),
            (0.55, 0.9),
            (0.65, 0.9),
            (0.65, 0.1),
            (0.75, 0.1),
            (0.75, 0.98),
            (0.45, 0.98),
        ]
        poly = fo.Polyline(points=[u_shape], filled=True)

        # by default, one shape per arm
        clipped = self._clip(poly)
        self.assertEqual(len(clipped.points), 2)
        self.assertEqual(
            sorted(round(_area(shape), 9) for shape in clipped.points),
            [0.25, 0.25],
        )
        xs = sorted(min(x for x, _ in shape) for shape in clipped.points)
        np.testing.assert_allclose(xs, [0.125, 0.625])

        # or one shape, whose arms are joined along the tile's border
        clipped = self._clip(poly, join_polygon_parts=True)
        self.assertEqual(len(clipped.points), 1)
        self.assertAlmostEqual(_area(clipped.points[0]), 0.5)

        # outside
        outside = [(0, 0), (0.1, 0), (0.1, 0.1)]
        self.assertIsNone(
            self._clip(fo.Polyline(points=[outside], filled=True))
        )

        # containing the tile
        around = [(0, 0), (1, 0), (1, 1), (0, 1)]
        clipped = self._clip(fo.Polyline(points=[around], filled=True))
        self.assertEqual(len(clipped.points), 1)
        self.assertAlmostEqual(_area(clipped.points[0]), 1)

    def _assert_same_inside(self, src_shapes, clipped, frame_size, tile):
        """Checks, on a grid of points of the tile, that the clipped shapes
        cover the same points as the source shapes, and that split parts do
        not overlap.
        """
        img_w, img_h = frame_size
        tile_x, tile_y, tile_w, tile_h = tile
        src_px = [[(x * img_w, y * img_h) for x, y in s] for s in src_shapes]
        for i in range(20):
            for j in range(20):
                # grid points that avoid the polygons' edges
                px = tile_x + (i + 0.5137) * tile_w / 20
                py = tile_y + (j + 0.4721) * tile_h / 20
                expected = any(_inside((px, py), s) for s in src_px)

                point = ((px - tile_x) / tile_w, (py - tile_y) / tile_h)
                shapes = clipped.points if clipped is not None else []
                num = sum(_inside(point, s) for s in shapes)
                self.assertEqual(num > 0, expected, msg=(i, j))
                self.assertLessEqual(num, 1, msg=(i, j))

    def test_random_polygons_keep_their_inside(self):
        rng = random.Random(0)
        for _ in range(200):
            # a random star-shaped polygon whose angular gaps are below 180
            # degrees, which makes it simple
            cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
            num = rng.randint(3, 12)
            angles = [
                2 * math.pi * (k + rng.uniform(0, 0.9)) / num
                for k in range(num)
            ]
            points = [
                (cx + r * math.cos(a), cy + r * math.sin(a))
                for a, r in ((a, rng.uniform(0.05, 0.5)) for a in angles)
            ]

            for join in (False, True):
                clipped = self._clip(
                    fo.Polyline(points=[points], filled=True),
                    join_polygon_parts=join,
                )
                if join and clipped is not None:
                    self.assertEqual(len(clipped.points), 1)

                self._assert_same_inside(
                    [points], clipped, self.SIZE, self.TILE
                )

    def test_polygons_with_bridged_holes(self):
        # Masks become polygons whose holes are bridged to their outer
        # boundary (by the eta fork, which runs them along it), and whose
        # thin parts visit the same vertices twice
        rng = np.random.default_rng(0)
        ys, xs = np.mgrid[0:200, 0:200]
        for _ in range(40):
            mask = np.zeros((200, 200), bool)
            for _ in range(rng.integers(1, 6)):
                cx, cy = rng.uniform(20, 180, 2)
                r_out = rng.uniform(15, 70)
                r_in = rng.uniform(3, max(4, r_out - 6))
                d = np.hypot(xs - cx, ys - cy)
                mask |= (d < r_out) & (d > r_in)

            cut = rng.uniform(0, 200, 2)
            mask &= np.hypot(xs - cut[0], ys - cut[1]) >= rng.uniform(5, 25)

            poly = fo.Polyline.from_mask(
                mask, label="m", tolerance=int(rng.integers(0, 3))
            )
            for _ in range(5):
                x, y = (int(v) for v in rng.integers(0, 180, 2))
                w, h = (
                    int(v) for v in rng.integers(5, 200 - max(x, y) + 1, 2)
                )
                for join in (False, True):
                    clipped = fout.clip_label(
                        poly, (x, y, w, h), (200, 200), join_polygon_parts=join
                    )
                    self._assert_same_inside(
                        poly.points, clipped, (200, 200), (x, y, w, h)
                    )

    def test_hole_whose_seam_crosses_the_tile_border(self):
        # a 20-180px square with a 80-120px square hole, bridged by a seam
        # at y=100, inside a tile at 50-150 x 60-140 that cuts the seam
        hole = [(80, 100), (80, 80), (120, 80), (120, 120), (80, 120)]
        for hole_points in (hole, hole[:1] + hole[:0:-1]):
            ring = [(20, 20), (180, 20), (180, 180), (20, 180), (20, 100)]
            ring += hole_points + [(80, 100), (20, 100)]
            poly = fo.Polyline(
                points=[[(x / 200, y / 200) for x, y in ring]], filled=True
            )

            tile = (50, 60, 100, 80)
            clipped = fout.clip_label(poly, tile, (200, 200))

            # the tile minus the hole, in one part
            self.assertEqual(len(clipped.points), 1)
            area = _area(clipped.points[0]) * 100 * 80
            self.assertAlmostEqual(area, 100 * 80 - 40 * 40)
            self._assert_same_inside(poly.points, clipped, (200, 200), tile)

            # as split, rather than through the fallback to joining
            parts = fout._clip_polygon_parts(
                fout._orient_loops(ring), (50, 60, 150, 140), 1e-7
            )
            self.assertEqual(len(parts), 1)
            self.assertAlmostEqual(_area(parts[0]), 100 * 80 - 40 * 40)

    def test_self_intersecting_polygons(self):
        # a bow tie keeps the points of its lobes
        bow_tie = [(0.3, 0.2), (0.7, 0.8), (0.7, 0.2), (0.3, 0.8)]
        for join in (False, True):
            clipped = self._clip(
                fo.Polyline(points=[bow_tie], filled=True),
                join_polygon_parts=join,
            )
            self._assert_same_inside([bow_tie], clipped, self.SIZE, self.TILE)

        # a polygon that crosses itself cannot be split into parts, so it is
        # clipped as with join_polygon_parts=True
        crossing = [
            (0.2596, 0.6734),
            (0.5745, 0.6759),
            (0.4570, 0.5501),
            (0.5786, 0.3926),
            (0.7091, 0.2589),
            (0.7303, 0.4137),
            (0.8549, 0.2985),
            (0.9000, 0.5495),
            (1.0249, 0.6189),
            (1.1495, 0.6546),
        ]
        poly = fo.Polyline(points=[crossing], filled=True)
        self.assertEqual(
            self._clip(poly).points,
            self._clip(poly, join_polygon_parts=True).points,
        )

    def test_polygons_inside_are_unchanged(self):
        triangle = [(0.5, 0.4), (0.6, 0.4), (0.55, 0.5)]
        clipped = self._clip(fo.Polyline(points=[triangle], filled=True))
        np.testing.assert_allclose(
            clipped.points[0], [(0.25, 1 / 3), (0.5, 1 / 3), (0.375, 0.5)]
        )

    def test_lines(self):
        # in, out, in again: two pieces
        line = [(0.5, 0.4), (0.5, 0.1), (0.6, 0.1), (0.6, 0.4)]
        clipped = self._clip(fo.Polyline(points=[line]))
        self.assertEqual(len(clipped.points), 2)
        np.testing.assert_allclose(
            clipped.points[0], [(0.25, 1 / 3), (0.25, 0)], atol=1e-12
        )
        np.testing.assert_allclose(
            clipped.points[1], [(0.5, 0), (0.5, 1 / 3)], atol=1e-12
        )

        # a closed outline that leaves the tile becomes open pieces
        ring = [(0.5, 0.4), (0.9, 0.4), (0.9, 0.6), (0.5, 0.6)]
        clipped = self._clip(fo.Polyline(points=[ring], closed=True))
        self.assertFalse(clipped.closed)
        self.assertEqual(len(clipped.points), 1)
        np.testing.assert_allclose(
            clipped.points[0],
            [(1, 2 / 3), (0.25, 2 / 3), (0.25, 1 / 3), (1, 1 / 3)],
            atol=1e-12,
        )

        # a closed outline inside stays closed
        ring = [(0.5, 0.4), (0.6, 0.4), (0.6, 0.6), (0.5, 0.6)]
        self.assertTrue(
            self._clip(fo.Polyline(points=[ring], closed=True)).closed
        )

        # unless another of its shapes is split, which opens the label, so
        # the outline inside repeats its first point
        split_ring = [(0.5, 0.4), (0.9, 0.4), (0.9, 0.6), (0.5, 0.6)]
        clipped = self._clip(
            fo.Polyline(points=[ring, split_ring], closed=True)
        )
        self.assertFalse(clipped.closed)
        inside = clipped.points[0]
        self.assertEqual(len(inside), 5)
        np.testing.assert_allclose(inside[0], inside[-1])

    def test_keypoints(self):
        kp = fo.Keypoint(
            label="person",
            points=[(0.5, 0.4), (0.1, 0.1), (nan, nan), (0.8, 0.8)],
            confidence=[0.9, 0.8, 0.7, 0.6],
        )
        clipped = self._clip(kp)

        # points outside become NaN, so the points keep their order
        self.assertEqual(len(clipped.points), 4)
        np.testing.assert_allclose(clipped.points[0], (0.25, 1 / 3))
        self.assertTrue(all(math.isnan(v) for v in clipped.points[1]))
        self.assertTrue(all(math.isnan(v) for v in clipped.points[2]))
        np.testing.assert_allclose(clipped.points[3], (1, 1))
        self.assertEqual(clipped.confidence, [0.9, 0.8, 0.7, 0.6])

        outside = fo.Keypoint(points=[(0.1, 0.1), (nan, nan)])
        self.assertIsNone(self._clip(outside))

    def test_segmentations_and_heatmaps(self):
        # at half the image's resolution
        mask = np.arange(250 * 500).reshape(250, 500).astype(np.int32)
        clipped = self._clip(fo.Segmentation(mask=mask))
        self.assertEqual(clipped.mask.shape, (150, 200))
        self.assertEqual(clipped.mask[0, 0], mask[50, 200])

        heatmap = fo.Heatmap(map=mask.astype(float), range=[0, 1])
        clipped = self._clip(heatmap)
        self.assertEqual(clipped.map.shape, (150, 200))
        self.assertEqual(clipped.map[-1, -1], mask[199, 399])
        self.assertEqual(clipped.range, [0, 1])

    def test_lists_and_other_labels(self):
        dets = fo.Detections(
            detections=[
                fo.Detection(label="in", bounding_box=[0.5, 0.4, 0.1, 0.1]),
                fo.Detection(label="out", bounding_box=[0, 0, 0.1, 0.1]),
            ]
        )
        clipped = self._clip(dets)
        self.assertEqual([d.label for d in clipped.detections], ["in"])
        self.assertEqual(len(dets.detections), 2)

        cls = fo.Classification(label="bridge")
        clipped = self._clip(cls)
        self.assertEqual(clipped.label, "bridge")
        self.assertNotEqual(clipped.id, cls.id)


class _TilesDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _dataset(self, images, **fields):
        """``images`` maps relative paths to image arrays."""
        dataset = fo.Dataset()
        samples = []
        for i, (relpath, img) in enumerate(images.items()):
            path = os.path.join(self.tmp, "src", relpath)
            _write_png(img, path)
            samples.append(
                fo.Sample(
                    filepath=path, **{k: v[i] for k, v in fields.items()}
                )
            )

        dataset.add_samples(samples)
        dataset.compute_metadata()
        return dataset


class MaterializeTilesTests(_TilesDatasetTests):
    @drop_datasets
    def test_crops_come_from_their_tiles(self):
        dataset = self._dataset({"a.png": _coords_image(1000, 500)})
        tiles = dataset.to_tiles((300, 200), overlap=50)
        out = os.path.join(self.tmp, "out")
        tiles_dataset = tiles.materialize(out)

        self.assertEqual(len(tiles_dataset), len(tiles))
        src = _coords_image(1000, 500)
        for sample in tiles_dataset:
            tile = sample.tile
            crop = _read(sample.filepath)
            self.assertTrue(sample.filepath.startswith(out + "/"))
            self.assertEqual(crop.shape, (tile.height, tile.width, 3))
            np.testing.assert_array_equal(
                crop,
                src[
                    tile.y : tile.y + tile.height, tile.x : tile.x + tile.width
                ],
            )
            self.assertEqual(
                (sample.metadata.width, sample.metadata.height),
                (tile.width, tile.height),
            )
            self.assertEqual(tile.sample_id, dataset.first().id)
            self.assertEqual(
                os.path.basename(sample.filepath),
                "a_tile_%d_%d.png" % (tile.row, tile.col),
            )

        # the last tile of each row is shifted back to the image's edge
        xs = sorted(set(tiles_dataset.values("tile.x")))
        self.assertEqual(xs, [0, 250, 500, 700])

    def _rotated_image(self, ext):
        """An image stored as 400x300 whose EXIF orientation rotates it 90
        degrees clockwise, to 300x400.
        """
        xs, ys = np.meshgrid(np.arange(400), np.arange(300))
        stored = np.stack(
            [xs * 255 // 399, ys * 255 // 299, np.zeros_like(xs)], axis=-1
        ).astype(np.uint8)

        exif = Image.Exif()
        exif[0x0112] = 6
        path = os.path.join(self.tmp, "src", "rotated" + ext)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(stored).save(path, exif=exif, quality=95)

        dataset = fo.Dataset()
        dataset.add_sample(fo.Sample(filepath=path))
        dataset.compute_metadata()
        metadata = dataset.first().metadata
        return dataset, path, (metadata.width, metadata.height)

    @drop_datasets
    def test_crops_follow_exif_orientation(self):
        # FiftyOne reads the EXIF orientation of JPEGs
        dataset, path, size = self._rotated_image(".jpg")
        self.assertEqual(size, (300, 400))

        with Image.open(path) as img:
            displayed = np.asarray(ImageOps.exif_transpose(img)).astype(int)

        tiles_dataset = dataset.to_tiles((300, 200)).materialize(
            os.path.join(self.tmp, "out")
        )
        self.assertEqual(len(tiles_dataset), 2)
        for sample in tiles_dataset:
            tile = sample.tile
            crop = _read(sample.filepath).astype(int)
            region = displayed[
                tile.y : tile.y + tile.height, tile.x : tile.x + tile.width
            ]

            # up to JPEG re-encoding
            self.assertEqual(crop.shape, region.shape)
            self.assertLess(np.abs(crop - region).mean(), 2)

    @drop_datasets
    def test_crops_follow_metadata_orientation(self):
        # FiftyOne does not read the EXIF orientation of PNGs, so their
        # tiles, and labels, are in their stored orientation
        dataset, path, size = self._rotated_image(".png")
        self.assertEqual(size, (400, 300))

        with Image.open(path) as img:
            stored = np.asarray(img)

        tiles_dataset = dataset.to_tiles((200, 300)).materialize(
            os.path.join(self.tmp, "out")
        )
        self.assertEqual(len(tiles_dataset), 2)
        for sample in tiles_dataset:
            tile = sample.tile
            np.testing.assert_array_equal(
                _read(sample.filepath),
                stored[
                    tile.y : tile.y + tile.height, tile.x : tile.x + tile.width
                ],
            )

    @drop_datasets
    def test_labels_are_clipped(self):
        # 600x500 tiles at x=0 and x=400
        dataset = self._dataset(
            {"a.png": _coords_image(1000, 500)},
            gt=[
                fo.Detections(
                    detections=[
                        fo.Detection(
                            label="left", bounding_box=[0.1, 0.1, 0.1, 0.1]
                        ),
                        fo.Detection(
                            label="both", bounding_box=[0.3, 0.1, 0.4, 0.1]
                        ),
                    ]
                )
            ],
            kps=[
                fo.Keypoints(
                    keypoints=[fo.Keypoint(points=[(0.2, 0.5), (0.9, 0.5)])]
                )
            ],
            weather=[fo.Classification(label="sunny")],
            seg=[
                fo.Segmentation(
                    mask=np.tile(np.arange(1000) // 4, (500, 1)).astype(
                        np.uint8
                    )
                )
            ],
            score=[0.5],
        )
        dataset.default_classes = ["left", "both"]
        dataset.mask_targets = {"seg": {1: "road"}}

        tiles = dataset.to_tiles((600, 600))
        left, right = list(tiles.materialize(os.path.join(self.tmp, "out")))

        self.assertEqual(
            [d.label for d in left.gt.detections], ["left", "both"]
        )
        _assert_box(left.gt.detections[1], [0.5, 0.1, 0.5, 0.1])
        self.assertEqual([d.label for d in right.gt.detections], ["both"])
        _assert_box(right.gt.detections[0], [0, 0.1, 0.5, 0.1])

        np.testing.assert_allclose(
            left.kps.keypoints[0].points[0], (1 / 3, 0.5)
        )
        self.assertTrue(math.isnan(left.kps.keypoints[0].points[1][0]))

        self.assertEqual(left.weather.label, "sunny")
        self.assertEqual(left.score, 0.5)
        self.assertEqual(left.seg.mask.shape, (500, 600))
        self.assertEqual(right.seg.mask[0, 0], 100)

        tiles_dataset = left._dataset
        self.assertEqual(tiles_dataset.default_classes, ["left", "both"])
        self.assertEqual(tiles_dataset.mask_targets, {"seg": {1: "road"}})

        # label IDs are new, since a label can be in several tiles
        self.assertNotEqual(
            left.gt.detections[1].id, right.gt.detections[0].id
        )

    @drop_datasets
    def test_filenames(self):
        img = _coords_image(500, 300)
        dataset = self._dataset({"x/img.png": img, "y/img.png": img})
        tiles = dataset.to_tiles((300, 300))

        # same basename: made unique
        names = sorted(
            os.path.basename(p)
            for p in tiles.materialize(os.path.join(self.tmp, "flat")).values(
                "filepath"
            )
        )
        self.assertEqual(
            names,
            [
                "img_tile_0_0-2.png",
                "img_tile_0_0.png",
                "img_tile_0_1-2.png",
                "img_tile_0_1.png",
            ],
        )

        # rel_dir keeps the source directories
        out = os.path.join(self.tmp, "nested")
        paths = tiles.materialize(
            out, rel_dir=os.path.join(self.tmp, "src")
        ).values("filepath")
        self.assertEqual(
            sorted(os.path.relpath(p, out) for p in paths),
            [
                "x/img_tile_0_0.png",
                "x/img_tile_0_1.png",
                "y/img_tile_0_0.png",
                "y/img_tile_0_1.png",
            ],
        )

        # image_format converts, e.g. RGBA to JPEG
        rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        dataset = self._dataset({"rgba.png": rgba})
        paths = (
            dataset.to_tiles((300, 300))
            .materialize(os.path.join(self.tmp, "jpg"), image_format=".jpg")
            .values("filepath")
        )
        self.assertTrue(all(p.endswith(".jpg") for p in paths))
        self.assertEqual(_read(paths[0]).shape, (300, 300, 3))

    @drop_datasets
    def test_masks_on_disk_go_to_the_output_dir(self):
        mask_path = os.path.join(self.tmp, "masks", "seg.png")
        _write_png(np.full((500, 1000), 7, np.uint8), mask_path)
        dataset = self._dataset(
            {"a.png": _coords_image(1000, 500)},
            seg=[fo.Segmentation(mask_path=mask_path)],
        )

        out = os.path.join(self.tmp, "out")
        tiles_dataset = dataset.to_tiles((600, 600)).materialize(out)
        for path in tiles_dataset.values("seg.mask_path"):
            self.assertTrue(
                path.startswith(os.path.join(out, "fields", "seg"))
            )
            mask = _read(path)
            self.assertEqual(mask.shape, (500, 600))
            self.assertTrue((mask == 7).all())

    @drop_datasets
    def test_source_is_not_modified(self):
        dataset = self._dataset(
            {"a.png": _coords_image(1000, 500)},
            gt=[
                fo.Detections(
                    detections=[
                        fo.Detection(bounding_box=[0.3, 0.1, 0.4, 0.1])
                    ]
                )
            ],
        )
        docs = list(dataset._sample_collection.find({}))
        dataset.to_tiles((600, 600)).materialize(os.path.join(self.tmp, "out"))
        self.assertEqual(list(dataset._sample_collection.find({})), docs)

    @drop_datasets
    def test_invalid_inputs(self):
        dataset = self._dataset(
            {"a.png": _coords_image(1000, 500)},
            tile=[1],
        )
        out = os.path.join(self.tmp, "out")

        with self.assertRaises(ValueError):
            fout.materialize_tiles(dataset, out)

        tiles = dataset.to_tiles((600, 600))
        for kwargs in (
            {},  # the source has a `tile` field
            {"tile_field": "src", "fields": ["missing"]},
            {"tile_field": "src", "fields": [fot.TILE_FIELD]},
        ):
            with self.assertRaises(ValueError, msg=kwargs):
                tiles.materialize(out, **kwargs)

        self.assertEqual(
            len(tiles.materialize(out, tile_field="src", fields=["tile"])), 2
        )

        # metadata that does not match the image
        dataset.set_values("metadata.width", [2000])
        with self.assertRaises(ValueError):
            dataset.to_tiles((600, 600)).materialize(out, tile_field="src")


class ExportTilesTests(_TilesDatasetTests):
    def _gt_dataset(self):
        # 600x500 tiles at x=0 and x=400
        return self._dataset(
            {
                "a.png": _coords_image(1000, 500),
                "b.png": _coords_image(1000, 500),
            },
            gt=[
                fo.Detections(
                    detections=[
                        fo.Detection(
                            label="cat", bounding_box=[0.1, 0.1, 0.1, 0.1]
                        ),
                        fo.Detection(
                            label="dog", bounding_box=[0.3, 0.2, 0.4, 0.2]
                        ),
                    ]
                ),
                fo.Detections(
                    detections=[
                        fo.Detection(
                            label="dog", bounding_box=[0.8, 0.6, 0.1, 0.2]
                        )
                    ]
                ),
            ],
        )

    @drop_datasets
    def test_yolov5_round_trip(self):
        dataset = self._gt_dataset()
        tiles = dataset.to_tiles((600, 600))
        expected = {
            os.path.splitext(os.path.basename(s.filepath))[0]: s
            for s in tiles.materialize(os.path.join(self.tmp, "expected"))
        }

        export_dir = os.path.join(self.tmp, "yolo")
        tiles.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="gt",
            classes=["cat", "dog"],
        )

        imported = fo.Dataset.from_dir(
            dataset_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="gt",
        )
        self.assertEqual(len(imported), 4)
        imported.compute_metadata()

        for sample in imported:
            name = os.path.splitext(os.path.basename(sample.filepath))[0]
            tile = expected[name]
            self.assertEqual(
                (sample.metadata.width, sample.metadata.height), (600, 500)
            )
            np.testing.assert_array_equal(
                _read(sample.filepath), _read(tile.filepath)
            )

            labels = [d.label for d in sample.gt.detections]
            self.assertEqual(labels, [d.label for d in tile.gt.detections])
            for actual, exp in zip(sample.gt.detections, tile.gt.detections):
                np.testing.assert_allclose(
                    actual.bounding_box, exp.bounding_box, atol=1e-5
                )

        # the clipped dog in the left tile of a.png
        dog = expected["a_tile_0_0"].gt.detections[1]
        _assert_box(dog, [0.5, 0.2, 0.5, 0.2])

    @drop_datasets
    def test_yolov5_splits(self):
        dataset = self._gt_dataset()
        dataset.take(1, seed=0).tag_samples("train")
        dataset.match_tags("train", bool=False).tag_samples("val")
        tiles = dataset.to_tiles((600, 600))

        export_dir = os.path.join(self.tmp, "yolo")
        for split in ("train", "val"):
            tiles.match_tags(split).export(
                export_dir=export_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="gt",
                split=split,
                classes=["cat", "dog"],
            )

        for split in ("train", "val"):
            num = len(os.listdir(os.path.join(export_dir, "images", split)))
            self.assertEqual(num, 2)

        with open(os.path.join(export_dir, "dataset.yaml")) as f:
            yaml = f.read()

        self.assertIn("train: ./images/train/", yaml)
        self.assertIn("val: ./images/val/", yaml)

    @drop_datasets
    def test_yolov5_polygons_and_masks(self):
        square = [(0.3, 0.2), (0.5, 0.2), (0.5, 0.4), (0.3, 0.4)]
        mask = np.ones((100, 200), bool)
        dataset = self._dataset(
            {"a.png": _coords_image(1000, 500)},
            polys=[
                fo.Polylines(
                    polylines=[
                        fo.Polyline(
                            label="lot",
                            points=[square],
                            closed=True,
                            filled=True,
                        )
                    ]
                )
            ],
            instances=[
                fo.Detections(
                    detections=[
                        fo.Detection(
                            label="car",
                            bounding_box=[0.3, 0.2, 0.2, 0.2],
                            mask=mask,
                        )
                    ]
                )
            ],
        )
        tiles = dataset.to_tiles((600, 600))

        for field, kwargs in (
            ("polys", {}),
            ("instances", {"use_masks": True}),
        ):
            export_dir = os.path.join(self.tmp, field)
            tiles.export(
                export_dir=export_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                label_field=field,
                **kwargs,
            )
            imported = fo.Dataset.from_dir(
                dataset_dir=export_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                label_type="polylines",
                label_field="polys",
            )

            widths = []
            for sample in imported:
                (shape,) = sample.polys.polylines[0].points
                xs, ys = zip(*shape)
                self.assertGreaterEqual(min(xs), 0)
                self.assertLessEqual(max(xs), 1)
                self.assertGreaterEqual(min(ys), 0)
                self.assertLessEqual(max(ys), 1)
                widths.append(max(xs) - min(xs))

            # the 300-500px square is inside the left tile (0-600px), and
            # clipped to 400-500px in the right one (400-1000px)
            # masks become polygons through pixel boundaries, which are up
            # to a pixel narrower
            atol = 2 / 600 if field == "instances" else 1e-6
            np.testing.assert_allclose(
                sorted(widths), [100 / 600, 200 / 600], atol=atol
            )

    def _u_dataset(self):
        """A 1500x1000 image cut into 3x2 tiles of 500px, with a U-shaped
        polygon whose bounding box covers the top middle tile, which the
        polygon does not enter, and an image with no labels at all.
        """
        u_shape = [
            (100, 100),
            (200, 100),
            (200, 900),
            (1300, 900),
            (1300, 100),
            (1400, 100),
            (1400, 950),
            (100, 950),
        ]
        poly = fo.Polyline(
            label="u",
            points=[[(x / 1500, y / 1000) for x, y in u_shape]],
            closed=True,
            filled=True,
        )
        return self._dataset(
            {
                "a.png": _coords_image(1500, 1000),
                "b.png": _coords_image(1500, 1000),
            },
            polys=[fo.Polylines(polylines=[poly]), None],
        )

    def _yolo_label_files(self, export_dir):
        labels_dir = os.path.join(export_dir, "labels", "val")
        rows = {}
        for filename in os.listdir(labels_dir):
            with open(os.path.join(labels_dir, filename)) as f:
                rows[os.path.splitext(filename)[0]] = f.read().splitlines()

        return rows

    @drop_datasets
    def test_empty_tiles(self):
        dataset = self._u_dataset()
        tiles = dataset.to_tiles((500, 500))
        self.assertEqual(len(tiles), 12)

        # the view keeps the polygon in the top middle tile of a.png, where
        # clipping removes it
        self.assertEqual(
            tiles.match(F("polys.polylines").length() > 0).count(), 6
        )
        labeled = [
            "a_tile_0_0",
            "a_tile_0_2",
            "a_tile_1_0",
            "a_tile_1_1",
            "a_tile_1_2",
        ]

        def export(empty_tiles):
            export_dir = os.path.join(self.tmp, empty_tiles)
            with self.assertLogs("fiftyone.utils.tiles", "INFO") as logs:
                tiles.export(
                    export_dir,
                    dataset_type=fo.types.YOLOv5Dataset,
                    label_field="polys",
                    empty_tiles=empty_tiles,
                )

            images = os.listdir(os.path.join(export_dir, "images", "val"))
            rows = self._yolo_label_files(export_dir)
            return logs.output[0], len(images), rows

        # with empty label files, including for b.png, which has no labels
        message, num_images, rows = export("keep")
        self.assertIn("Exporting 7 of 12 tile(s) without labels", message)
        self.assertEqual(num_images, 12)
        self.assertEqual(len(rows), 12)
        self.assertEqual(sorted(k for k, v in rows.items() if v), labeled)
        self.assertEqual(rows["a_tile_0_1"], [])
        self.assertEqual(rows["b_tile_0_0"], [])

        # without label files
        message, num_images, rows = export("keep_without_labels")
        self.assertIn("Exporting 7 of 12 tile(s) without labels", message)
        self.assertIn("no label files", message)
        self.assertEqual(num_images, 12)
        self.assertEqual(sorted(rows), labeled)

        # not at all
        message, num_images, rows = export("skip")
        self.assertIn("Skipping 7 of 12 tile(s)", message)
        self.assertEqual(num_images, 5)
        self.assertEqual(sorted(rows), labeled)
        self.assertTrue(all(rows.values()))

        # keep is the default
        export_dir = os.path.join(self.tmp, "default")
        tiles.export(
            export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="polys",
        )
        self.assertEqual(len(self._yolo_label_files(export_dir)), 12)

        for kwargs in (
            {"label_field": "polys", "empty_tiles": "drop"},
            {"empty_tiles": "skip"},
            {"empty_tiles": "keep_without_labels"},
        ):
            with self.assertRaises(ValueError, msg=kwargs):
                tiles.export(
                    os.path.join(self.tmp, "bad"),
                    dataset_type=fo.types.YOLOv5Dataset,
                    **kwargs,
                )

    @drop_datasets
    def test_polygon_parts(self):
        dataset = self._u_dataset()
        tiles = dataset.to_tiles((500, 500)).match(
            F("filepath").ends_with("a.png")
        )

        # the bottom middle tile cuts the U's bar off its arms: the bar is
        # one part, but each side tile holds an arm and part of the bar
        export_dir = os.path.join(self.tmp, "split")
        tiles.export(
            export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="polys",
        )
        rows = self._yolo_label_files(export_dir)
        self.assertEqual(len(rows["a_tile_1_1"]), 1)

        # the top tiles of the arms each hold one arm
        self.assertEqual(len(rows["a_tile_0_0"]), 1)

        # a 1500x500 crop of the U's top, as one tile, cuts it into its two
        # arms: one shape, or row, per arm by default
        tall = dataset.to_tiles((1500, 500)).match(
            F("filepath").ends_with("a.png")
        )
        export_dir = os.path.join(self.tmp, "arms")
        tall.export(
            export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="polys",
        )
        self.assertEqual(
            len(self._yolo_label_files(export_dir)["a_tile_0_0"]), 2
        )

        export_dir = os.path.join(self.tmp, "joined")
        tall.export(
            export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="polys",
            join_polygon_parts=True,
        )
        self.assertEqual(
            len(self._yolo_label_files(export_dir)["a_tile_0_0"]), 1
        )

        # materialize() takes the same option
        shapes = tall.materialize(os.path.join(self.tmp, "m1")).values(
            "polys.polylines.points"
        )
        self.assertEqual(len(shapes[0][0]), 2)
        shapes = tall.materialize(
            os.path.join(self.tmp, "m2"), join_polygon_parts=True
        ).values("polys.polylines.points")
        self.assertEqual(len(shapes[0][0]), 1)

    @drop_datasets
    def test_any_format(self):
        dataset = self._gt_dataset()
        tiles = dataset.to_tiles((600, 600))

        # this used to raise an IndexError
        export_dir = os.path.join(self.tmp, "images")
        tiles.export(export_dir, dataset_type=fo.types.ImageDirectory)
        self.assertEqual(len(os.listdir(export_dir)), 4)

        export_dir = os.path.join(self.tmp, "coco")
        tiles.export(
            export_dir,
            dataset_type=fo.types.COCODetectionDataset,
            label_field="gt",
        )
        imported = fo.Dataset.from_dir(
            dataset_dir=export_dir,
            dataset_type=fo.types.COCODetectionDataset,
            label_types="detections",
            label_field="gt",
        )
        self.assertEqual(len(imported), 4)
        self.assertEqual(
            sorted(len(s.gt.detections) if s.gt else 0 for s in imported),
            [0, 1, 1, 2],
        )

    @drop_datasets
    def test_export_arguments(self):
        dataset = self._gt_dataset()
        tiles = dataset.to_tiles((600, 600))
        export_dir = os.path.join(self.tmp, "yolo")

        for export_media in (False, "symlink", "manifest"):
            with self.assertRaises(ValueError):
                tiles.export(
                    export_dir,
                    dataset_type=fo.types.YOLOv5Dataset,
                    label_field="gt",
                    export_media=export_media,
                )

        with self.assertRaises(ValueError):
            tiles.export(
                labels_path=os.path.join(self.tmp, "labels.json"),
                dataset_type=fo.types.COCODetectionDataset,
                label_field="gt",
            )

        # an exporter instance
        exporter = fouy.YOLOv5DatasetExporter(export_dir)
        tiles.export(dataset_exporter=exporter, label_field="gt")
        self.assertEqual(
            len(os.listdir(os.path.join(export_dir, "images", "val"))), 4
        )

    @drop_datasets
    def test_temporary_data_is_deleted(self):
        dataset = self._gt_dataset()
        tiles = dataset.to_tiles((600, 600))
        datasets = set(fo.list_datasets())
        tmp_dirs = []
        make_temp_dir = fout.etau.make_temp_dir

        def _make_temp_dir(*args, **kwargs):
            tmp_dirs.append(make_temp_dir(*args, **kwargs))
            return tmp_dirs[-1]

        fout.etau.make_temp_dir = _make_temp_dir
        try:
            tiles.export(
                os.path.join(self.tmp, "yolo"),
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="gt",
            )
            with self.assertRaises(Exception):
                tiles.export(
                    os.path.join(self.tmp, "bad"),
                    dataset_type=fo.types.YOLOv5Dataset,
                    label_field="missing",
                )
        finally:
            fout.etau.make_temp_dir = make_temp_dir

        self.assertEqual(len(tmp_dirs), 2)
        self.assertFalse(any(os.path.exists(d) for d in tmp_dirs))
        self.assertEqual(set(fo.list_datasets()), datasets)


class SplitTests(_TilesDatasetTests):
    _LABELS = ("cat", "dog", "bird", "fish", "cow")

    def _split_dataset(self):
        """Five 1000x500 images, cut into 600x500 tiles at x=0 and x=400,
        each with one label in its left tile: two tagged ``train``, two
        ``val``, and one with no split.
        """
        return self._dataset(
            {
                "img%d.png" % i: _coords_image(1000, 500)
                for i in range(len(self._LABELS))
            },
            gt=[
                fo.Detections(
                    detections=[
                        fo.Detection(
                            label=label, bounding_box=[0.1, 0.1, 0.1, 0.2]
                        )
                    ]
                )
                for label in self._LABELS
            ],
            tags=[["train"], ["train"], ["val"], ["val", "other"], []],
            split=["a", "b", "a", None, "b"],
        )

    def _image_stems(self, export_dir):
        """The stems of the source images of the exported tile images."""
        return {
            f.rsplit("_tile_", 1)[0]
            for f in os.listdir(export_dir)
            if f.endswith(".png")
        }

    @drop_datasets
    def test_get_splits_by_tags(self):
        dataset = self._split_dataset()
        ids = dataset.values("id")
        tiles = dataset.to_tiles((600, 600))

        splits = fout.get_splits(tiles, ["train", "val"])
        self.assertEqual(list(splits), ["train", "val"])
        self.assertEqual(sorted(splits["train"]), sorted(ids[:2]))
        self.assertEqual(sorted(splits["val"]), sorted(ids[2:4]))

        # all the tags, by default
        splits = fout.get_splits(tiles, split_field="tags")
        self.assertEqual(list(splits), ["other", "train", "val"])

        # an image with two of the tags is in both splits
        dataset.select(ids[0]).tag_samples("val")
        splits = fout.get_splits(tiles, ["train", "val"])
        self.assertIn(ids[0], splits["train"])
        self.assertIn(ids[0], splits["val"])

        # only the images of the view
        view = tiles.match(F("filepath").ends_with("img2.png"))
        splits = fout.get_splits(view, ["train", "val"])
        self.assertEqual(splits, {"train": [], "val": [ids[2]]})

    @drop_datasets
    def test_get_splits_by_field(self):
        dataset = self._split_dataset()
        ids = dataset.values("id")
        tiles = dataset.to_tiles((600, 600))

        splits = fout.get_splits(tiles, split_field="split")
        self.assertEqual(list(splits), ["a", "b"])
        self.assertEqual(sorted(splits["a"]), sorted([ids[0], ids[2]]))
        self.assertEqual(sorted(splits["b"]), sorted([ids[1], ids[4]]))

        splits = fout.get_splits(tiles, ["b"], split_field="split")
        self.assertEqual(list(splits), ["b"])

    @drop_datasets
    def test_random_split(self):
        dataset = self._split_dataset()
        ids = dataset.values("id")
        tiles = dataset.to_tiles((600, 600))

        fracs = {"train": 0.6, "val": 0.4}
        splits = fout.get_splits(tiles, fracs, seed=1)
        self.assertEqual(list(splits), ["train", "val"])
        self.assertEqual([len(v) for v in splits.values()], [3, 2])
        self.assertEqual(sorted(splits["train"] + splits["val"]), sorted(ids))

        # reproducible, whatever the order of the tiles
        self.assertEqual(fout.get_splits(tiles, fracs, seed=1), splits)
        reverse = tiles.sort_by("filepath", reverse=True)
        self.assertEqual(fout.get_splits(reverse, fracs, seed=1), splits)

        # the fractions are normalized, and may be zero
        splits = fout.get_splits(
            tiles, {"train": 3, "val": 2, "test": 0}, seed=1
        )
        self.assertEqual([len(v) for v in splits.values()], [3, 2, 0])

    @drop_datasets
    def test_invalid_splits(self):
        dataset = self._split_dataset()
        tiles = dataset.to_tiles((600, 600))

        for kwargs in (
            {"splits": {"train": 1}, "split_field": "tags"},
            {"splits": {"train": -1, "val": 1}},
            {"splits": {"train": 0}},
            {"splits": []},
            {"splits": ["train", "train"]},
            {"splits": ["a/b"]},
            {"split_field": "gt"},
            {"split_field": "missing"},
        ):
            with self.assertRaises(ValueError, msg=kwargs):
                fout.get_splits(tiles, **kwargs)

    @drop_datasets
    def test_export_yolov5_splits(self):
        dataset = self._split_dataset()
        tiles = dataset.to_tiles((600, 600))
        export_dir = os.path.join(self.tmp, "yolo")

        # the export directory is deleted once, before the first split
        stale = os.path.join(export_dir, "stale.txt")
        os.makedirs(export_dir)
        open(stale, "w").close()

        counts = tiles.export(
            export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="gt",
            splits=["train", "val"],
            overwrite=True,
        )

        self.assertEqual(dict(counts), {"train": 4, "val": 4})
        self.assertFalse(os.path.exists(stale))
        for split, stems in (
            ("train", {"img0", "img1"}),
            ("val", {"img2", "img3"}),
        ):
            images_dir = os.path.join(export_dir, "images", split)
            self.assertEqual(self._image_stems(images_dir), stems)
            self.assertEqual(len(os.listdir(images_dir)), 4)

        # the splits share the classes, so that they agree on the indices
        for split, labels in (
            ("train", {"cat", "dog"}),
            ("val", {"bird", "fish"}),
        ):
            imported = fo.Dataset.from_dir(
                dataset_dir=export_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                split=split,
                label_field="gt",
            )
            self.assertEqual(
                set(imported.distinct("gt.detections.label")), labels
            )

        # negative tiles are skipped in every split
        counts = tiles.export(
            os.path.join(self.tmp, "yolo_skip"),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="gt",
            splits=["train", "val"],
            empty_tiles="skip",
        )
        self.assertEqual(dict(counts), {"train": 2, "val": 2})

    @drop_datasets
    def test_export_splits_in_directories(self):
        dataset = self._split_dataset()
        tiles = dataset.to_tiles((600, 600))
        export_dir = os.path.join(self.tmp, "coco")

        counts = tiles.export(
            export_dir,
            dataset_type=fo.types.COCODetectionDataset,
            label_field="gt",
            splits={"train": 0.6, "val": 0.4},
            seed=1,
        )

        self.assertEqual(dict(counts), {"train": 6, "val": 4})
        self.assertEqual(sorted(os.listdir(export_dir)), ["train", "val"])

        stems = []
        categories = []
        for split in ("train", "val"):
            split_dir = os.path.join(export_dir, split)
            stems.append(self._image_stems(os.path.join(split_dir, "data")))
            with open(os.path.join(split_dir, "labels.json")) as f:
                categories.append(json.load(f)["categories"])

        # all the tiles of an image are in the same split, and the splits
        # share the categories
        self.assertEqual(len(stems[0] | stems[1]), 5)
        self.assertFalse(stems[0] & stems[1])
        self.assertEqual(categories[0], categories[1])
        self.assertEqual(len(categories[0]), 5)

    @drop_datasets
    def test_export_fiftyone_dataset_splits(self):
        dataset = self._split_dataset()
        tiles = dataset.to_tiles((600, 600))
        export_dir = os.path.join(self.tmp, "fo")

        counts = tiles.export(
            export_dir,
            dataset_type=fo.types.FiftyOneDataset,
            split_field="split",
        )

        # the image without a value has no split
        self.assertEqual(dict(counts), {"a": 4, "b": 4})
        imported = fo.Dataset.from_dir(
            dataset_dir=export_dir, dataset_type=fo.types.FiftyOneDataset
        )
        self.assertEqual(len(imported), 8)
        self.assertEqual(len(imported.match_tags("a")), 4)
        self.assertEqual(len(imported.match_tags("b")), 4)
        self.assertEqual(len(imported.match_tags(["a", "b"], all=True)), 0)

    @drop_datasets
    def test_split_export_errors(self):
        dataset = self._split_dataset()
        tiles = dataset.to_tiles((600, 600))
        export_dir = os.path.join(self.tmp, "yolo")
        kwargs = dict(
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="gt",
            splits=["train", "val"],
        )

        for extra in (
            {"export_dir": export_dir, "split": "train"},
            {"export_dir": export_dir, "data_path": "images"},
            {"labels_path": os.path.join(self.tmp, "labels")},
        ):
            with self.assertRaises(ValueError, msg=extra):
                tiles.export(**extra, **kwargs)

        exporter = fouy.YOLOv5DatasetExporter(export_dir)
        with self.assertRaises(ValueError):
            tiles.export(
                dataset_exporter=exporter,
                label_field="gt",
                splits=["train", "val"],
            )

        # an image cannot be in two splits
        dataset.select(dataset.first().id).tag_samples("val")
        with self.assertRaises(ValueError):
            tiles.export(export_dir, **kwargs)

        self.assertFalse(os.path.exists(export_dir))


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _without_ids(d):
    """A label dict without IDs, whose NaNs, plain or in extended JSON, are
    comparable.
    """
    if isinstance(d, dict):
        if d == {"$numberDouble": "NaN"}:
            return "nan"

        return {k: _without_ids(v) for k, v in d.items() if k != "_id"}

    if isinstance(d, list):
        return [_without_ids(v) for v in d]

    if isinstance(d, float) and math.isnan(d):
        return "nan"

    return d


def _visible_points(points):
    return [p for p in points if not any(math.isnan(v) for v in p)]


class ExportFormatsTests(_TilesDatasetTests):
    """Exports a 1000x500 image cut into 600x500 tiles at x=0 and x=400 to
    each image format, and reads it back with FiftyOne's importers.
    """

    def _tiles(self):
        nan_free_mask = np.ones((100, 400), dtype=bool)
        dataset = self._dataset(
            {"a.png": _coords_image(1000, 500)},
            gt=[
                fo.Detections(
                    detections=[
                        fo.Detection(
                            label="cat", bounding_box=[0.1, 0.1, 0.1, 0.2]
                        ),
                        # cut by both tiles, 3/4 of it inside each
                        fo.Detection(
                            label="dog", bounding_box=[0.3, 0.2, 0.4, 0.2]
                        ),
                    ]
                )
            ],
            instances=[
                fo.Detections(
                    detections=[
                        fo.Detection(
                            label="dog",
                            bounding_box=[0.3, 0.2, 0.4, 0.2],
                            mask=nan_free_mask,
                        )
                    ]
                )
            ],
            polys=[
                fo.Polylines(
                    polylines=[
                        fo.Polyline(
                            label="lot",
                            points=[
                                [
                                    (0.3, 0.6),
                                    (0.5, 0.6),
                                    (0.5, 0.8),
                                    (0.3, 0.8),
                                ]
                            ],
                            closed=True,
                            filled=True,
                        )
                    ]
                )
            ],
            kps=[
                fo.Keypoints(
                    keypoints=[
                        fo.Keypoint(
                            label="person",
                            points=[(0.2, 0.5), (0.5, 0.5), (0.9, 0.5)],
                        )
                    ]
                )
            ],
            seg=[
                fo.Segmentation(
                    mask=np.tile(np.arange(1000) // 100, (500, 1)).astype(
                        np.uint8
                    )
                )
            ],
            weather=[fo.Classification(label="sunny")],
            location=[fo.GeoLocation(point=[-73.9855, 40.758])],
            score=[0.5],
        )
        tiles = dataset.to_tiles((600, 600))
        expected = {
            _stem(s.filepath): s
            for s in tiles.materialize(os.path.join(self.tmp, "expected"))
        }
        return tiles, expected

    def _round_trip(
        self, tiles, expected, dataset_type, export_kwargs=None, **kwargs
    ):
        export_dir = os.path.join(self.tmp, dataset_type.__name__)
        tiles.export(
            export_dir, dataset_type=dataset_type, **(export_kwargs or {})
        )
        imported = fo.Dataset.from_dir(
            dataset_dir=export_dir, dataset_type=dataset_type, **kwargs
        )
        samples = {_stem(s.filepath): s for s in imported}
        self.assertEqual(sorted(samples), sorted(expected))
        return samples

    def _assert_detections(self, samples, expected, field, tol_px, src="gt"):
        for stem, sample in samples.items():
            actual = sample[field].detections
            wanted = expected[stem][src].detections
            self.assertEqual(
                [d.label for d in actual], [d.label for d in wanted]
            )
            for a, e in zip(actual, wanted):
                np.testing.assert_allclose(
                    a.bounding_box, e.bounding_box, atol=tol_px / 500
                )

    def _assert_images(self, samples, expected):
        for stem, sample in samples.items():
            np.testing.assert_array_equal(
                _read(sample.filepath), _read(expected[stem].filepath)
            )

    @drop_datasets
    def test_fiftyone_image_detection(self):
        tiles, expected = self._tiles()
        samples = self._round_trip(
            tiles,
            expected,
            fo.types.FiftyOneImageDetectionDataset,
            export_kwargs={"label_field": "gt"},
        )
        self._assert_detections(samples, expected, "ground_truth", 1e-6)
        self._assert_images(samples, expected)

    @drop_datasets
    def test_coco(self):
        tiles, expected = self._tiles()

        samples = self._round_trip(
            tiles,
            expected,
            fo.types.COCODetectionDataset,
            export_kwargs={"label_field": "gt"},
            label_types="detections",
        )
        self._assert_detections(samples, expected, "ground_truth", 1e-6)
        self._assert_images(samples, expected)

        # masks, cropped to the visible 300x100px of the dog
        samples = self._round_trip(
            tiles,
            expected,
            fo.types.COCODetectionDataset,
            export_kwargs={"label_field": "instances"},
            label_types="segmentations",
        )
        self._assert_detections(
            samples, expected, "ground_truth", 1e-6, src="instances"
        )
        for sample in samples.values():
            (det,) = sample.ground_truth.detections
            self.assertEqual(det.mask.shape, (100, 300))
            self.assertGreater(det.mask.mean(), 0.95)

        # polygons
        samples = self._round_trip(
            tiles,
            expected,
            fo.types.COCODetectionDataset,
            export_kwargs={"label_field": "polys"},
            label_types="segmentations",
            use_polylines=True,
        )
        for stem, sample in samples.items():
            (actual,) = sample.ground_truth.polylines
            (wanted,) = expected[stem].polys.polylines
            np.testing.assert_allclose(
                sorted(actual.points[0]),
                sorted(wanted.points[0]),
                atol=1 / 500,
            )

        # keypoints, whose points outside the tile stay hidden
        samples = self._round_trip(
            tiles,
            expected,
            fo.types.COCODetectionDataset,
            export_kwargs={"label_field": "kps"},
            label_types="keypoints",
        )
        for stem, sample in samples.items():
            (actual,) = sample.ground_truth.keypoints
            (wanted,) = expected[stem].kps.keypoints
            np.testing.assert_allclose(
                actual.points, wanted.points, atol=1 / 500
            )

    @drop_datasets
    def test_voc(self):
        tiles, expected = self._tiles()
        samples = self._round_trip(
            tiles,
            expected,
            fo.types.VOCDetectionDataset,
            export_kwargs={"label_field": "gt"},
        )
        self._assert_detections(samples, expected, "ground_truth", 1)
        self._assert_images(samples, expected)

        # VOC flags the boxes that the tiles cut
        truncated = {
            stem: [
                (d.label, d.get_attribute_value("truncated", None))
                for d in s.ground_truth.detections
            ]
            for stem, s in samples.items()
        }
        self.assertEqual(
            truncated,
            {
                "a_tile_0_0": [("cat", None), ("dog", 1)],
                "a_tile_0_1": [("dog", 1)],
            },
        )

    @drop_datasets
    def test_kitti(self):
        tiles, expected = self._tiles()
        samples = self._round_trip(
            tiles,
            expected,
            fo.types.KITTIDetectionDataset,
            export_kwargs={"label_field": "gt"},
        )
        self._assert_detections(samples, expected, "ground_truth", 1)
        self._assert_images(samples, expected)

        # KITTI has the fraction of each box outside the tile
        truncated = {
            stem: [
                (d.label, d.get_attribute_value("truncated", None))
                for d in s.ground_truth.detections
            ]
            for stem, s in samples.items()
        }
        self.assertEqual(
            truncated,
            {
                "a_tile_0_0": [("cat", 0.0), ("dog", 0.25)],
                "a_tile_0_1": [("dog", 0.25)],
            },
        )

    @drop_datasets
    def test_yolov4(self):
        tiles, expected = self._tiles()
        samples = self._round_trip(
            tiles,
            expected,
            fo.types.YOLOv4Dataset,
            export_kwargs={"label_field": "gt", "classes": ["cat", "dog"]},
        )
        self._assert_detections(samples, expected, "ground_truth", 1e-3)
        self._assert_images(samples, expected)

    @drop_datasets
    def test_cvat_image(self):
        tiles, expected = self._tiles()
        samples = self._round_trip(
            tiles,
            expected,
            fo.types.CVATImageDataset,
            export_kwargs={
                "label_field": {
                    "gt": "detections",
                    "polys": "polylines",
                    "kps": "keypoints",
                    "weather": "classifications",
                }
            },
        )
        self._assert_detections(samples, expected, "detections", 1)
        self._assert_images(samples, expected)
        for stem, sample in samples.items():
            self.assertEqual(
                sample.classifications.classifications[0].label, "sunny"
            )

            (actual,) = sample.polylines.polylines
            (wanted,) = expected[stem].polys.polylines
            np.testing.assert_allclose(
                sorted(actual.points[0]),
                sorted(wanted.points[0]),
                atol=1 / 500,
            )

            # CVAT points cannot be hidden, so only the visible ones remain
            (actual,) = sample.keypoints.keypoints
            (wanted,) = expected[stem].kps.keypoints
            np.testing.assert_allclose(
                actual.points, _visible_points(wanted.points), atol=1 / 500
            )

    @drop_datasets
    def test_image_segmentation_directory(self):
        tiles, expected = self._tiles()
        samples = self._round_trip(
            tiles,
            expected,
            fo.types.ImageSegmentationDirectory,
            export_kwargs={"label_field": "seg"},
        )
        self._assert_images(samples, expected)
        for stem, sample in samples.items():
            np.testing.assert_array_equal(
                sample.ground_truth.get_mask(), expected[stem].seg.get_mask()
            )

        # x=400 is in class 4
        self.assertEqual(
            samples["a_tile_0_1"].ground_truth.get_mask()[0, 0], 4
        )

    @drop_datasets
    def test_fiftyone_image_labels(self):
        tiles, expected = self._tiles()
        samples = self._round_trip(
            tiles,
            expected,
            fo.types.FiftyOneImageLabelsDataset,
            export_kwargs={
                "label_field": {
                    "gt": "detections",
                    "polys": "polylines",
                    "kps": "keypoints",
                }
            },
        )
        self._assert_detections(samples, expected, "detections", 1e-6)
        self._assert_images(samples, expected)
        for stem, sample in samples.items():
            np.testing.assert_allclose(
                sample.polylines.polylines[0].points,
                expected[stem].polys.polylines[0].points,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                sample.keypoints.keypoints[0].points,
                expected[stem].kps.keypoints[0].points,
                atol=1e-9,
            )

    @drop_datasets
    def test_bdd(self):
        tiles, expected = self._tiles()
        samples = self._round_trip(
            tiles,
            expected,
            fo.types.BDDDataset,
            export_kwargs={
                "label_field": {"gt": "detections", "polys": "polylines"}
            },
        )
        self._assert_detections(samples, expected, "detections", 0.1)
        self._assert_images(samples, expected)
        for stem, sample in samples.items():
            np.testing.assert_allclose(
                sorted(sample.polylines.polylines[0].points[0]),
                sorted(expected[stem].polys.polylines[0].points[0]),
                atol=0.1 / 500,
            )

    @drop_datasets
    def test_classification_formats(self):
        tiles, expected = self._tiles()
        for dataset_type in (
            fo.types.FiftyOneImageClassificationDataset,
            fo.types.ImageClassificationDirectoryTree,
        ):
            samples = self._round_trip(
                tiles,
                expected,
                dataset_type,
                export_kwargs={"label_field": "weather"},
            )
            self._assert_images(samples, expected)
            self.assertEqual(
                [s.ground_truth.label for s in samples.values()],
                ["sunny", "sunny"],
            )

    @drop_datasets
    def test_media_formats(self):
        tiles, expected = self._tiles()
        for dataset_type in (fo.types.ImageDirectory, fo.types.MediaDirectory):
            samples = self._round_trip(tiles, expected, dataset_type)
            self._assert_images(samples, expected)

    @drop_datasets
    def test_csv_and_geojson(self):
        tiles, expected = self._tiles()

        samples = self._round_trip(
            tiles,
            expected,
            fo.types.CSVDataset,
            export_kwargs={"fields": ["filepath", "score"]},
        )
        self._assert_images(samples, expected)
        self.assertEqual([s.score for s in samples.values()], ["0.5", "0.5"])

        samples = self._round_trip(tiles, expected, fo.types.GeoJSONDataset)
        self._assert_images(samples, expected)
        for sample in samples.values():
            self.assertEqual(sample.location.point, [-73.9855, 40.758])

    @drop_datasets
    def test_fiftyone_formats(self):
        tiles, expected = self._tiles()
        for dataset_type in (
            fo.types.FiftyOneDataset,
            fo.types.LegacyFiftyOneDataset,
        ):
            samples = self._round_trip(tiles, expected, dataset_type)
            self._assert_images(samples, expected)
            for stem, sample in samples.items():
                wanted = expected[stem]
                self.assertEqual(sample.tile.to_dict(), wanted.tile.to_dict())

                # each export materializes the tiles again, with new IDs
                for field in ("gt", "polys", "kps"):
                    self.assertEqual(
                        _without_ids(sample[field].to_dict()),
                        _without_ids(wanted[field].to_dict()),
                    )

                np.testing.assert_array_equal(
                    sample.instances.detections[0].mask,
                    wanted.instances.detections[0].mask,
                )
                np.testing.assert_array_equal(
                    sample.seg.get_mask(), wanted.seg.get_mask()
                )

    @drop_datasets
    def test_tf_formats(self):
        try:
            import tensorflow  # pylint: disable=unused-import
        except ImportError:
            self.skipTest("TensorFlow is not installed")

        tiles, expected = self._tiles()
        expected = [expected[stem] for stem in sorted(expected)]

        def round_trip(dataset_type, label_field):
            export_dir = os.path.join(self.tmp, dataset_type.__name__)
            tiles.export(
                export_dir,
                dataset_type=dataset_type,
                label_field=label_field,
                image_format=".png",
            )

            # TFRecords do not keep the image filenames, but their order
            imported = fo.Dataset.from_dir(
                dataset_dir=export_dir,
                dataset_type=dataset_type,
                images_dir=os.path.join(export_dir, "images"),
                image_format=".png",
            )
            samples = sorted(imported, key=lambda s: s.filepath)
            self.assertEqual(len(samples), len(expected))
            for sample, wanted in zip(samples, expected):
                np.testing.assert_array_equal(
                    _read(sample.filepath), _read(wanted.filepath)
                )

            return zip(samples, expected)

        for sample, wanted in round_trip(
            fo.types.TFObjectDetectionDataset, "gt"
        ):
            actual = sample.ground_truth.detections
            self.assertEqual(
                [d.label for d in actual],
                [d.label for d in wanted.gt.detections],
            )
            for a, e in zip(actual, wanted.gt.detections):
                np.testing.assert_allclose(
                    a.bounding_box, e.bounding_box, atol=1e-6
                )

        for sample, _ in round_trip(
            fo.types.TFImageClassificationDataset, "weather"
        ):
            self.assertEqual(sample.ground_truth.label, "sunny")


if __name__ == "__main__":
    fo.config.show_progress_bars = False
    unittest.main(verbosity=2)

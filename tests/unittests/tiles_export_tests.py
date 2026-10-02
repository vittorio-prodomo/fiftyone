"""
FiftyOne tiles materialization and export unit tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import math
import os
import random
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageOps

import fiftyone as fo
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

        # a U that the tile cuts into two parts stays one polygon, joined
        # along the tile's border
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
        clipped = self._clip(fo.Polyline(points=[u_shape], filled=True))
        self.assertEqual(len(clipped.points), 1)

        # the parts inside the tile (y < 400px) are the two 100px wide arms
        arm = 0.25 * 1.0
        self.assertAlmostEqual(_area(clipped.points[0]), 2 * arm)

        # outside
        outside = [(0, 0), (0.1, 0), (0.1, 0.1)]
        self.assertIsNone(
            self._clip(fo.Polyline(points=[outside], filled=True))
        )

    def test_random_polygons_keep_their_inside(self):
        rng = random.Random(0)
        tile_x, tile_y, tile_w, tile_h = self.TILE
        for _ in range(100):
            # a random star-shaped polygon, which is simple
            cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
            num = rng.randint(3, 12)
            angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(num))
            radii = [rng.uniform(0.05, 0.5) for _ in range(num)]
            points = [
                (cx + r * math.cos(a), cy + r * math.sin(a))
                for a, r in zip(angles, radii)
            ]

            clipped = self._clip(fo.Polyline(points=[points], filled=True))

            src_px = [(x * 1000, y * 500) for x, y in points]
            for i in range(20):
                for j in range(20):
                    # grid points that avoid the polygons' edges
                    px = tile_x + (i + 0.5137) * tile_w / 20
                    py = tile_y + (j + 0.4721) * tile_h / 20
                    expected = _inside((px, py), src_px)

                    if clipped is None:
                        self.assertFalse(expected)
                        continue

                    point = ((px - tile_x) / tile_w, (py - tile_y) / tile_h)
                    actual = any(_inside(point, s) for s in clipped.points)
                    self.assertEqual(actual, expected, msg=(points, i, j))

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


if __name__ == "__main__":
    fo.config.show_progress_bars = False
    unittest.main(verbosity=2)

"""
FiftyOne tiles view unit tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import random
import re
import subprocess
import sys
import unittest

import fiftyone as fo
import fiftyone.core.stages as fos
import fiftyone.core.tiles as fot
from fiftyone import ViewField as F

from decorators import drop_datasets


def _box_px(tile):
    return (tile["x"], tile["y"], tile["width"], tile["height"])


def _tiles_px(img_w, img_h, tile_size, overlap=0):
    return [
        _box_px(t) for t in fot.compute_tiles(img_w, img_h, tile_size, overlap)
    ]


def _describe(schema):
    return {
        name: (type(field), getattr(field, "document_type", None))
        for name, field in schema.items()
    }


def _make_dataset(sizes, **fields):
    dataset = fo.Dataset()
    dataset.add_samples(
        [
            fo.Sample(
                filepath="/tmp/tile%d.jpg" % i,
                metadata=fo.ImageMetadata(width=w, height=h),
                **{k: v[i] for k, v in fields.items()},
            )
            for i, (w, h) in enumerate(sizes)
        ]
    )
    return dataset


def _labels(tile, field):
    value = tile[field]
    if value is None:
        return None

    if isinstance(value, fo.Detections):
        return sorted(d.label for d in value.detections)

    if isinstance(value, fo.Polylines):
        return sorted(p.label for p in value.polylines)

    if isinstance(value, fo.Keypoints):
        return sorted(k.label for k in value.keypoints)

    return value.label


class ComputeTilesTests(unittest.TestCase):
    def test_exact_fit(self):
        self.assertEqual(
            _tiles_px(1280, 640, (640, 640)),
            [(0, 0, 640, 640), (640, 0, 640, 640)],
        )

    def test_last_tile_is_shifted_back_to_the_edge(self):
        self.assertEqual(
            _tiles_px(1000, 640, (640, 640)),
            [(0, 0, 640, 640), (360, 0, 640, 640)],
        )

    def test_uses_the_minimum_number_of_tiles(self):
        # stride 128: the 4th column already reaches the edge
        xs = [x for x, _, _, _ in _tiles_px(1000, 640, (640, 640), 512)]
        self.assertEqual(xs, [0, 128, 256, 360])

    def test_overlap_in_pixels_and_fraction(self):
        self.assertEqual(
            _tiles_px(1000, 100, (400, 400), overlap=100),
            _tiles_px(1000, 100, (400, 400), overlap=0.25),
        )
        xs = [x for x, _, _, _ in _tiles_px(1000, 100, (400, 400), 100)]
        self.assertEqual(xs, [0, 300, 600])

    def test_tiles_are_clamped_where_the_image_is_smaller(self):
        self.assertEqual(
            _tiles_px(1000, 500, (600, 600), overlap=100),
            [(0, 0, 600, 500), (400, 0, 600, 500)],
        )
        self.assertEqual(_tiles_px(300, 200, (640, 640)), [(0, 0, 300, 200)])

    def test_rows_and_cols(self):
        tiles = fot.compute_tiles(1000, 1000, (640, 640))
        self.assertEqual(
            [(t["row"], t["col"]) for t in tiles],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )

    def test_random_grids_cover_the_image_with_full_tiles(self):
        rng = random.Random(0)
        for _ in range(300):
            img_w, img_h = rng.randint(1, 3000), rng.randint(1, 3000)
            tile_w, tile_h = rng.randint(10, 800), rng.randint(10, 800)
            overlap = rng.choice([0, rng.random() * 0.9])
            tiles = fot.compute_tiles(img_w, img_h, (tile_w, tile_h), overlap)

            xs = sorted({t["x"] for t in tiles})
            ys = sorted({t["y"] for t in tiles})
            _, _, overlap_x, overlap_y = fot._parse_tiling(
                (tile_w, tile_h), overlap
            )

            self.assertEqual(len(tiles), len(xs) * len(ys))
            for t in tiles:
                self.assertEqual(t["width"], min(tile_w, img_w))
                self.assertEqual(t["height"], min(tile_h, img_h))
                self.assertLessEqual(t["x"] + t["width"], img_w)
                self.assertLessEqual(t["y"] + t["height"], img_h)

            # the grid starts at 0, ends at the edge, and every gap is at
            # most the stride
            for starts, img_len, tile_len, ov in (
                (xs, img_w, tile_w, overlap_x),
                (ys, img_h, tile_h, overlap_y),
            ):
                self.assertEqual(starts[0], 0)
                self.assertEqual(starts[-1] + min(tile_len, img_len), img_len)
                gaps = [b - a for a, b in zip(starts, starts[1:])]
                self.assertTrue(all(0 < g <= tile_len - ov for g in gaps))

                # minimal: one tile fewer, even at the full stride, would not
                # reach the edge
                if len(starts) > 1:
                    stride = tile_len - ov
                    self.assertLess(
                        (len(starts) - 2) * stride + tile_len, img_len
                    )

    def test_invalid_arguments(self):
        for tile_size, overlap in (
            ((0, 640), 0),
            ((640,), 0),
            ((640.5, 640), 0),
            ((640, 640), -1),
            ((640, 640), 640),
            ((640, 100), 100),
        ):
            with self.assertRaises(ValueError):
                fot.compute_tiles(1000, 1000, tile_size, overlap)

        with self.assertRaises(ValueError):
            fot.compute_tiles(0, 1000, (640, 640))

    def test_compute_tile_detections(self):
        dets = fot.compute_tile_detections(1000, 500, (600, 600), overlap=100)
        self.assertEqual([d.label for d in dets], ["tile_0_0", "tile_0_1"])
        self.assertEqual(dets[1].bounding_box, [0.4, 0.0, 0.6, 1.0])


class TilesViewTests(unittest.TestCase):
    @drop_datasets
    def test_matches_the_reference_grid(self):
        rng = random.Random(1)
        sizes = [
            (rng.randint(1, 2500), rng.randint(1, 2500)) for _ in range(60)
        ]
        dataset = _make_dataset(sizes)

        for tile_size, overlap in (((640, 640), 0), ((500, 300), 0.2)):
            view = dataset.to_tiles(tile_size, overlap=overlap)
            by_sample = {}
            for sample_id, box in zip(
                view.values("sample_id"),
                view.values("tile_regions.bounding_box"),
            ):
                by_sample.setdefault(sample_id, []).append(box)

            for sample in dataset:
                w, h = sample.metadata.width, sample.metadata.height
                expected = _tiles_px(w, h, tile_size, overlap)
                actual = [
                    (
                        round(x * w),
                        round(y * h),
                        round(bw * w),
                        round(bh * h),
                    )
                    for x, y, bw, bh in by_sample[sample.id]
                ]
                self.assertEqual(sorted(actual), sorted(expected))

    @drop_datasets
    def test_view_basics(self):
        dataset = _make_dataset([(1000, 500)])
        view = dataset.to_tiles((600, 600), overlap=100)

        self.assertIsInstance(view, fot.TilesView)
        self.assertEqual(view.patches_field, "tile_regions")
        self.assertEqual(len(view), 2)
        self.assertEqual(
            view.values("tile_regions.label"), ["tile_0_0", "tile_0_1"]
        )
        self.assertEqual(set(view.values("sample_id")), {dataset.first().id})
        self.assertEqual(view.values("id"), view.values("tile_regions.id"))
        self.assertEqual(view.distinct("metadata.width"), [1000])

    @drop_datasets
    def test_missing_metadata_raises(self):
        dataset = _make_dataset([(1000, 500)])
        dataset.add_sample(fo.Sample(filepath="/tmp/no_metadata.jpg"))
        with self.assertRaises(ValueError):
            dataset.to_tiles((600, 600))

    @drop_datasets
    def test_other_fields(self):
        dataset = _make_dataset(
            [(1000, 500)],
            gt=[fo.Detections()],
            weather=["rain"],
        )

        view = dataset.to_tiles((600, 600))
        self.assertIn("gt", view.get_field_schema())
        self.assertEqual(view.distinct("weather"), ["rain"])

        view = dataset.to_tiles((600, 600), other_fields=False)
        self.assertNotIn("gt", view.get_field_schema())
        self.assertNotIn("weather", view.get_field_schema())

        view = dataset.to_tiles((600, 600), other_fields="weather")
        self.assertNotIn("gt", view.get_field_schema())
        self.assertIn("weather", view.get_field_schema())

    @drop_datasets
    def test_serialization(self):
        stage = fo.ToTiles((600, 600), overlap=0.1, min_label_coverage=0.5)
        stage2 = fos.ViewStage._from_dict(stage._serialize())
        self.assertEqual(stage2.tile_size, (600, 600))
        self.assertEqual(stage2.overlap, 0.1)
        self.assertEqual(stage2.min_label_coverage, 0.5)

        # views saved before edge tiles were shifted still load
        d = stage._serialize()
        d["kwargs"].append(["min_coverage", 0.3])
        self.assertIsNone(fos.ViewStage._from_dict(d).config)

    @drop_datasets
    def test_saved_view(self):
        dataset = _make_dataset([(1000, 500), (700, 700)])
        dataset.save_view("tiles", dataset.to_tiles((600, 600)))

        view = dataset.load_saved_view("tiles")
        self.assertIsInstance(view, fot.TilesView)
        self.assertEqual(len(view), 6)

        view.reload()
        self.assertEqual(len(view), 6)


class TilesLabelFilterTests(unittest.TestCase):
    """A 1000x500 image cut into 600x500 tiles at x=0 and x=400."""

    def _tiles(self, min_label_coverage=0.0, **fields):
        dataset = _make_dataset(
            [(1000, 500)], **{k: [v] for k, v in fields.items()}
        )
        view = dataset.to_tiles(
            (600, 600), overlap=100, min_label_coverage=min_label_coverage
        )
        return list(view)

    @drop_datasets
    def test_detections(self):
        gt = fo.Detections(
            detections=[
                fo.Detection(label="left", bounding_box=[0.05, 0.1, 0.1, 0.2]),
                fo.Detection(label="both", bounding_box=[0.45, 0.1, 0.1, 0.2]),
                fo.Detection(
                    label="right", bounding_box=[0.85, 0.1, 0.1, 0.2]
                ),
            ]
        )
        left, right = self._tiles(gt=gt)
        self.assertEqual(_labels(left, "gt"), ["both", "left"])
        self.assertEqual(_labels(right, "gt"), ["both", "right"])

        # labels keep their full-image coordinates
        self.assertEqual(
            left.gt.detections[0].bounding_box, [0.05, 0.1, 0.1, 0.2]
        )

    @drop_datasets
    def test_touching_an_edge_is_not_overlapping(self):
        # ends exactly where the right tile starts (x=0.4)
        gt = fo.Detections(
            detections=[
                fo.Detection(label="a", bounding_box=[0.3, 0, 0.1, 0.1])
            ]
        )
        left, right = self._tiles(gt=gt)
        self.assertEqual(_labels(left, "gt"), ["a"])
        self.assertEqual(_labels(right, "gt"), [])

    @drop_datasets
    def test_touching_an_edge_despite_float_rounding(self):
        # x=1px, w=21px ends at 22px, where the second tile starts, but
        # 0.001 + 0.021 == 0.022000000000000002 > 0.022
        dataset = _make_dataset(
            [(1000, 500)],
            gt=[
                fo.Detections(
                    detections=[
                        fo.Detection(
                            label="a", bounding_box=[0.001, 0.1, 0.021, 0.1]
                        )
                    ]
                )
            ],
        )
        view = dataset.to_tiles((22, 500))
        first, second = view.limit(2)
        self.assertEqual(_labels(first, "gt"), ["a"])
        self.assertEqual(_labels(second, "gt"), [])

    @drop_datasets
    def test_min_label_coverage(self):
        # spans x=0.5-0.7: half of it lies in the left tile (x < 0.6), all
        # of it in the right tile (x >= 0.4)
        gt = fo.Detections(
            detections=[
                fo.Detection(label="a", bounding_box=[0.5, 0.1, 0.2, 0.2])
            ]
        )
        left, right = self._tiles(min_label_coverage=0.6, gt=gt)
        self.assertEqual(_labels(left, "gt"), [])
        self.assertEqual(_labels(right, "gt"), ["a"])

        left, right = self._tiles(min_label_coverage=0.5, gt=gt)
        self.assertEqual(_labels(left, "gt"), ["a"])

        with self.assertRaises(ValueError):
            self._tiles(min_label_coverage=1.5, gt=gt)

    @drop_datasets
    def test_polylines_use_the_bounds_of_their_points(self):
        polylines = fo.Polylines(
            polylines=[
                fo.Polyline(label="left", points=[[(0.1, 0.1), (0.2, 0.3)]]),
                fo.Polyline(
                    label="two_shapes",
                    points=[
                        [(0.1, 0.1), (0.15, 0.1)],
                        [(0.9, 0.5), (0.95, 0.6)],
                    ],
                ),
                # horizontal segment: zero area, still crosses both tiles
                fo.Polyline(
                    label="segment", points=[[(0.3, 0.5), (0.7, 0.5)]]
                ),
            ]
        )
        left, right = self._tiles(lines=polylines)
        self.assertEqual(
            _labels(left, "lines"), ["left", "segment", "two_shapes"]
        )
        self.assertEqual(_labels(right, "lines"), ["segment", "two_shapes"])

    @drop_datasets
    def test_keypoints_ignore_hidden_points(self):
        nan = float("nan")
        kps = fo.Keypoints(
            keypoints=[
                fo.Keypoint(label="left", points=[(0.1, 0.1), (nan, nan)]),
                fo.Keypoint(label="right", points=[(nan, nan), (0.9, 0.5)]),
                fo.Keypoint(label="hidden", points=[(nan, nan)]),
            ]
        )
        left, right = self._tiles(kps=kps)
        self.assertEqual(_labels(left, "kps"), ["left"])
        self.assertEqual(_labels(right, "kps"), ["right"])

    @drop_datasets
    def test_single_label_fields(self):
        left, right = self._tiles(
            det=fo.Detection(label="d", bounding_box=[0.05, 0.1, 0.1, 0.1]),
            line=fo.Polyline(label="p", points=[[(0.9, 0.1), (0.95, 0.2)]]),
            kp=fo.Keypoint(label="k", points=[(0.5, 0.5)]),
            cls=fo.Classification(label="bridge"),
        )
        self.assertEqual(_labels(left, "det"), "d")
        self.assertIsNone(left.line)
        self.assertIsNone(right.det)
        self.assertEqual(_labels(right, "line"), "p")
        self.assertEqual(_labels(left, "kp"), "k")
        self.assertEqual(_labels(right, "kp"), "k")

        # image-level labels are copied as-is
        self.assertEqual(left.cls.label, "bridge")
        self.assertEqual(right.cls.label, "bridge")

    @drop_datasets
    def test_empty_fields_stay_empty(self):
        dataset = _make_dataset(
            [(1000, 500), (1000, 500)],
            gt=[None, fo.Detections()],
        )
        view = dataset.to_tiles((600, 600), overlap=100)
        values = view.values("gt")
        self.assertEqual(values[:2], [None, None])
        self.assertEqual([len(v.detections) for v in values[2:]], [0, 0])


def _reference_counts(sizes, boxes_px, tile_size, overlap):
    """Counts the boxes that no tile contains by checking every tile."""
    count = uncontained = too_large = 0
    for (img_w, img_h), boxes in zip(sizes, boxes_px):
        tiles = fot.compute_tiles(img_w, img_h, tile_size, overlap)
        tile_w, tile_h = tiles[0]["width"], tiles[0]["height"]
        for x, y, w, h in boxes:
            # only the part inside the image counts
            x1, x2 = min(max(x, 0), img_w), min(max(x + w, 0), img_w)
            y1, y2 = min(max(y, 0), img_h), min(max(y + h, 0), img_h)
            count += 1
            if any(
                t["x"] <= x1
                and x2 <= t["x"] + tile_w
                and t["y"] <= y1
                and y2 <= t["y"] + tile_h
                for t in tiles
            ):
                continue

            uncontained += 1
            if x2 - x1 > tile_w or y2 - y1 > tile_h:
                too_large += 1

    return {"count": count, "uncontained": uncontained, "too_large": too_large}


def _detections(img_w, img_h, boxes_px):
    return fo.Detections(
        detections=[
            fo.Detection(
                label="obj",
                bounding_box=[x / img_w, y / img_h, w / img_w, h / img_h],
            )
            for x, y, w, h in boxes_px
        ]
    )


class CountUncontainedLabelsTests(unittest.TestCase):
    """A 1000x500 image cut into 600x500 tiles at x=0 and x=400."""

    def _counts(self, tile_size=(600, 600), overlap=100, **fields):
        dataset = _make_dataset(
            [(1000, 500)], **{k: [v] for k, v in fields.items()}
        )
        return fot.count_uncontained_labels(
            dataset, tile_size, overlap=overlap
        )

    @drop_datasets
    def test_matches_the_reference(self):
        rng = random.Random(0)
        for _ in range(40):
            tile_size = (rng.randint(20, 300), rng.randint(20, 300))
            overlap = rng.choice(
                [0, rng.randint(1, min(tile_size) - 1), rng.random() * 0.9]
            )
            sizes = [
                (rng.randint(10, 800), rng.randint(10, 800)) for _ in range(4)
            ]

            boxes_px = []
            for img_w, img_h in sizes:
                tiles = fot.compute_tiles(img_w, img_h, tile_size, overlap)
                boxes = []
                for _ in range(40):
                    if rng.random() < 0.5:
                        # snapped to tile edges, where float rounding matters
                        t = rng.choice(tiles)
                        x = t["x"] + rng.choice([-1, 0, 0, 1])
                        y = t["y"] + rng.choice([-1, 0, 0, 1])
                        w = t["width"] + rng.choice([-1, 0, 0, 1]) - x + t["x"]
                        h = (
                            t["height"]
                            + rng.choice([-1, 0, 0, 1])
                            - y
                            + t["y"]
                        )
                        w, h = max(w, 0), max(h, 0)
                    else:
                        x = rng.randint(-20, img_w)
                        y = rng.randint(-20, img_h)
                        w = rng.randint(0, img_w // 2)
                        h = rng.randint(0, img_h // 2)

                    boxes.append((x, y, w, h))

                boxes_px.append(boxes)

            dataset = _make_dataset(
                sizes,
                gt=[
                    _detections(img_w, img_h, boxes)
                    for (img_w, img_h), boxes in zip(sizes, boxes_px)
                ],
            )
            counts = fot.count_uncontained_labels(
                dataset, tile_size, overlap=overlap
            )
            self.assertEqual(
                counts["gt"],
                _reference_counts(sizes, boxes_px, tile_size, overlap),
                msg="tile_size=%s, overlap=%s, sizes=%s"
                % (tile_size, overlap, sizes),
            )
            dataset.delete()

    @drop_datasets
    def test_fits_crosses_and_too_large(self):
        gt = _detections(
            1000,
            500,
            [
                (50, 50, 100, 100),  # in the left tile
                (400, 0, 600, 500),  # exactly the right tile
                (350, 50, 300, 100),  # crosses both tiles' borders
                (500, 50, 100, 100),  # in the overlap
                (100, 50, 700, 100),  # wider than a tile
            ],
        )
        counts = self._counts(gt=gt)
        self.assertEqual(
            counts, {"gt": {"count": 5, "uncontained": 2, "too_large": 1}}
        )

    @drop_datasets
    def test_labels_no_larger_than_the_overlap_always_fit(self):
        # 600x500 tiles at x=0 and x=500, which overlap by exactly 100px
        dataset = _make_dataset(
            [(1100, 500)],
            fits=[
                _detections(
                    1100, 500, [(x, 50, 100, 100) for x in range(1001)]
                )
            ],
            crosses=[_detections(1100, 500, [(499.5, 50, 101, 100)])],
        )
        counts = fot.count_uncontained_labels(dataset, (600, 600), overlap=100)
        self.assertEqual(counts["fits"]["uncontained"], 0)
        self.assertEqual(counts["crosses"]["uncontained"], 1)

    @drop_datasets
    def test_fits_despite_float_rounding(self):
        # x=1px, w=21px ends at 22px, where the first tile ends, but
        # 0.001 + 0.021 == 0.022000000000000002 > 0.022
        gt = fo.Detections(
            detections=[
                fo.Detection(label="a", bounding_box=[0.001, 0.1, 0.021, 0.1])
            ]
        )
        counts = self._counts(tile_size=(22, 500), overlap=0, gt=gt)
        self.assertEqual(counts["gt"]["uncontained"], 0)

    @drop_datasets
    def test_only_the_part_inside_the_image_counts(self):
        gt = _detections(
            1000, 500, [(-100, -50, 200, 100), (950, 450, 100, 100)]
        )
        counts = self._counts(gt=gt)
        self.assertEqual(counts["gt"]["uncontained"], 0)

    @drop_datasets
    def test_polylines_and_keypoints_use_the_bounds_of_their_points(self):
        nan = float("nan")
        counts = self._counts(
            lines=fo.Polylines(
                polylines=[
                    # each shape fits in a tile, but not both
                    fo.Polyline(
                        points=[[(0.35, 0.1), (0.38, 0.1)], [(0.65, 0.2)]]
                    ),
                    fo.Polyline(points=[[(0.45, 0.1), (0.55, 0.2)]]),
                ]
            ),
            kps=fo.Keypoints(
                keypoints=[
                    # the hidden point would put it across both tiles
                    fo.Keypoint(points=[(0.05, 0.1), (nan, nan), (0.1, 0.2)]),
                    fo.Keypoint(points=[(0.05, 0.1), (0.9, 0.2)]),
                    # all points hidden: no box
                    fo.Keypoint(points=[(nan, nan)]),
                ]
            ),
        )
        self.assertEqual(
            counts,
            {
                "lines": {"count": 2, "uncontained": 1, "too_large": 0},
                "kps": {"count": 2, "uncontained": 1, "too_large": 1},
            },
        )

    @drop_datasets
    def test_single_label_fields_and_missing_boxes(self):
        dataset = _make_dataset(
            [(1000, 500), (1000, 500), (1000, 500)],
            det=[
                fo.Detection(bounding_box=[0.35, 0.1, 0.3, 0.1]),
                fo.Detection(label="no box"),
                None,
            ],
            gt=[
                fo.Detections(
                    detections=[
                        fo.Detection(bounding_box=[0.05, 0.1, 0.1, 0.1]),
                        fo.Detection(label="no box"),
                    ]
                ),
                None,
                fo.Detections(),
            ],
        )
        counts = fot.count_uncontained_labels(dataset, (600, 600), overlap=100)
        self.assertEqual(
            counts,
            {
                "det": {"count": 1, "uncontained": 1, "too_large": 0},
                "gt": {"count": 1, "uncontained": 0, "too_large": 0},
            },
        )

    @drop_datasets
    def test_fields(self):
        dataset = _make_dataset(
            [(1000, 500)],
            gt=[_detections(1000, 500, [(350, 50, 300, 100)])],
            pred=[_detections(1000, 500, [(50, 50, 100, 100)])],
            cls=[fo.Classification(label="bridge")],
        )

        # all supported fields by default
        counts = fot.count_uncontained_labels(dataset, (600, 600), 100)
        self.assertEqual(list(counts), ["gt", "pred"])

        counts = fot.count_uncontained_labels(
            dataset, (600, 600), 100, fields="pred"
        )
        self.assertEqual(list(counts), ["pred"])

        for fields in ("cls", ["gt", "missing"]):
            with self.assertRaises(ValueError):
                fot.count_uncontained_labels(
                    dataset, (600, 600), 100, fields=fields
                )

        # the labels of the collection, not of its dataset
        view = dataset.filter_labels("gt", F("label") == "other")
        counts = fot.count_uncontained_labels(view, (600, 600), 100)
        self.assertEqual(counts["gt"]["count"], 0)

        # no supported fields
        counts = fot.count_uncontained_labels(
            dataset.exclude_fields(["gt", "pred"]), (600, 600), 100
        )
        self.assertEqual(counts, {})

    @drop_datasets
    def test_missing_metadata_raises(self):
        dataset = _make_dataset([(1000, 500)])
        dataset.add_sample(fo.Sample(filepath="/tmp/nometa.jpg"))

        with self.assertRaises(ValueError):
            fot.count_uncontained_labels(dataset, (600, 600))


class TilesSourceSafetyTests(unittest.TestCase):
    """Tiles views are views: the source collection is never modified."""

    def _raw_docs(self, dataset):
        return list(dataset._sample_collection.find({}, sort=[("_id", 1)]))

    @drop_datasets
    def test_source_is_not_modified(self):
        dataset = _make_dataset(
            [(1000, 500), (700, 700)],
            gt=[
                fo.Detections(
                    detections=[
                        fo.Detection(
                            label="a", bounding_box=[0.1, 0.1, 0.1, 0.1]
                        )
                    ]
                ),
                None,
            ],
        )
        docs = self._raw_docs(dataset)
        schema = _describe(dataset.get_field_schema())
        last_modified_at = dataset.last_modified_at

        view = dataset.to_tiles((600, 600), overlap=100)
        view.tag_samples("tile_tag")
        view.tag_labels("tile_label_tag", label_fields="tile_regions")
        view.tag_labels("gt_tag", label_fields="gt")

        # the tags live in the view...
        self.assertEqual(view.count_sample_tags(), {"tile_tag": len(view)})
        self.assertEqual(
            view.count_label_tags("tile_regions"),
            {"tile_label_tag": len(view)},
        )

        # ...and nowhere else
        dataset.reload()
        self.assertEqual(self._raw_docs(dataset), docs)
        self.assertEqual(_describe(dataset.get_field_schema()), schema)
        self.assertEqual(dataset.last_modified_at, last_modified_at)

    @drop_datasets
    def test_existing_tile_regions_field_is_left_alone(self):
        dataset = _make_dataset(
            [(1000, 500)],
            tile_regions=[
                fo.Detections(detections=[fo.Detection(label="mine")])
            ],
        )
        docs = self._raw_docs(dataset)

        view = dataset.to_tiles((600, 600))
        self.assertEqual(
            view.values("tile_regions.label"), ["tile_0_0", "tile_0_1"]
        )
        self.assertEqual(self._raw_docs(dataset), docs)

        with self.assertRaises(ValueError):
            dataset.to_tiles((600, 600), other_fields=["tile_regions"])

    @drop_datasets
    def test_tiles_of_a_patches_view(self):
        # patches views have their own sample_id field
        dataset = _make_dataset(
            [(1000, 500)],
            gt=[
                fo.Detections(
                    detections=[
                        fo.Detection(label="a", bounding_box=[0, 0, 0.5, 0.5])
                    ]
                )
            ],
        )
        patches = dataset.to_patches("gt")
        view = patches.to_tiles((600, 600))
        self.assertEqual(set(view.values("sample_id")), {patches.first().id})

        with self.assertRaises(ValueError):
            patches.to_tiles((600, 600), other_fields=["sample_id"])

    @drop_datasets
    def test_concurrent_views_from_separate_processes(self):
        dataset = _make_dataset([(1000 + i, 700) for i in range(300)])
        dataset.persistent = True

        script = (
            "import sys; import fiftyone as fo; "
            "fo.config.show_progress_bars = False; "
            "d = fo.load_dataset(sys.argv[1]); "
            "print(len(d.to_tiles((int(sys.argv[2]), int(sys.argv[2])))))"
        )
        expected = {
            size: len(dataset.to_tiles((size, size))) for size in (256, 128)
        }

        try:
            procs = {
                size: subprocess.Popen(
                    [sys.executable, "-c", script, dataset.name, str(size)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for size in expected
            }
            for size, proc in procs.items():
                out, err = proc.communicate(timeout=300)
                self.assertEqual(proc.returncode, 0, err[-2000:])
                self.assertEqual(
                    int(out.strip().splitlines()[-1]), expected[size]
                )
        finally:
            dataset.delete()


class TilesAppTests(unittest.TestCase):
    """The App hardcodes a few names of the Python tiles implementation."""

    def _read_app_source(self, *path):
        path = os.path.join(
            os.path.dirname(os.path.dirname(fo.__file__)), "app", *path
        )
        if not os.path.isfile(path):
            self.skipTest("App sources not available")

        with open(path) as f:
            return f.read()

    def test_app_uses_the_same_tile_field(self):
        source = self._read_app_source("packages", "looker", "src", "zoom.ts")
        match = re.search(r'TILES_FIELD = "([^"]+)"', source)

        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), fot.TILE_FIELD)
        self.assertEqual(fo.ToTiles._TILE_FIELD, fot.TILE_FIELD)

    def test_app_treats_tiles_views_as_patches_views(self):
        source = self._read_app_source(
            "packages", "state", "src", "recoil", "view.ts"
        )
        match = re.search(r'TILES_VIEW = "([^"]+)"', source)

        self.assertIsNotNone(match)
        self.assertEqual(
            match.group(1),
            fot.TilesView.__module__ + "." + fot.TilesView.__name__,
        )


if __name__ == "__main__":
    fo.config.show_progress_bars = False
    unittest.main(verbosity=2)

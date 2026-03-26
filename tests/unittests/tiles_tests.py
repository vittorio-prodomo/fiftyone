"""
FiftyOne tiles-related unit tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import unittest

import fiftyone as fo
import fiftyone.core.patches as fop

from decorators import drop_datasets
from fiftyone.core.tiles import compute_tile_detections


class ComputeTileDetectionsTests(unittest.TestCase):
    def test_exact_fit_no_overlap(self):
        """640x640 tiles on a 1280x1280 image => 4 tiles, no overlap."""
        dets = compute_tile_detections(
            img_w=1280, img_h=1280, tile_w=640, tile_h=640, overlap=0
        )
        self.assertEqual(len(dets), 4)
        self._assert_bbox(dets[0], 0.0, 0.0, 0.5, 0.5)
        self._assert_bbox(dets[1], 0.5, 0.0, 0.5, 0.5)
        self._assert_bbox(dets[2], 0.0, 0.5, 0.5, 0.5)
        self._assert_bbox(dets[3], 0.5, 0.5, 0.5, 0.5)

    def test_non_divisible_produces_edge_tiles(self):
        """1000x1000 image with 640x640 tiles => 4 tiles, edge tiles are smaller."""
        dets = compute_tile_detections(
            img_w=1000, img_h=1000, tile_w=640, tile_h=640, overlap=0
        )
        self.assertEqual(len(dets), 4)
        self.assertAlmostEqual(
            dets[1].bounding_box[0] + dets[1].bounding_box[2], 1.0, places=6
        )

    def test_overlap_pixels(self):
        """Overlap specified in pixels (>= 1)."""
        dets = compute_tile_detections(
            img_w=1280, img_h=1280, tile_w=640, tile_h=640, overlap=64
        )
        # stride=576, starts: 0, 576, 1152 => 3x3 = 9
        self.assertEqual(len(dets), 9)

    def test_overlap_fraction(self):
        """Overlap < 1 treated as fraction of tile size."""
        dets = compute_tile_detections(
            img_w=1280, img_h=1280, tile_w=640, tile_h=640, overlap=0.1
        )
        # 0.1 * 640 = 64px overlap, same as above
        self.assertEqual(len(dets), 9)

    def test_min_coverage_filters_small_edge_tiles(self):
        """Edge tiles below min_coverage are excluded."""
        dets_all = compute_tile_detections(
            img_w=1000,
            img_h=1000,
            tile_w=640,
            tile_h=640,
            overlap=0,
            min_coverage=0.0,
        )
        dets_filtered = compute_tile_detections(
            img_w=1000,
            img_h=1000,
            tile_w=640,
            tile_h=640,
            overlap=0,
            min_coverage=0.5,
        )
        self.assertLess(len(dets_filtered), len(dets_all))

    def test_tile_labels_contain_row_col(self):
        """Each detection label encodes row/col position."""
        dets = compute_tile_detections(
            img_w=1280, img_h=1280, tile_w=640, tile_h=640, overlap=0
        )
        self.assertEqual(dets[0].label, "tile_0_0")
        self.assertEqual(dets[1].label, "tile_0_1")
        self.assertEqual(dets[2].label, "tile_1_0")

    def _assert_bbox(self, det, x, y, w, h, places=6):
        bb = det.bounding_box
        self.assertAlmostEqual(bb[0], x, places=places)
        self.assertAlmostEqual(bb[1], y, places=places)
        self.assertAlmostEqual(bb[2], w, places=places)
        self.assertAlmostEqual(bb[3], h, places=places)


class ToTilesTests(unittest.TestCase):
    @drop_datasets
    def test_to_tiles_basic(self):
        """Basic tiling: 2x2 grid on 1280x1280 images."""
        dataset = fo.Dataset()
        sample = fo.Sample(
            filepath="image1.png",
            metadata=fo.ImageMetadata(width=1280, height=1280),
            ground_truth=fo.Detections(
                detections=[
                    fo.Detection(
                        label="cat", bounding_box=[0.1, 0.1, 0.3, 0.3]
                    ),
                ]
            ),
        )
        dataset.add_sample(sample)
        view = dataset.to_tiles(tile_size=(640, 640))
        self.assertEqual(len(view), 4)
        tile = view.first()
        self.assertIn("ground_truth", tile.field_names)

    @drop_datasets
    def test_to_tiles_with_overlap(self):
        dataset = fo.Dataset()
        sample = fo.Sample(
            filepath="image1.png",
            metadata=fo.ImageMetadata(width=1280, height=1280),
        )
        dataset.add_sample(sample)
        view = dataset.to_tiles(tile_size=(640, 640), overlap=64)
        self.assertEqual(len(view), 9)

    @drop_datasets
    def test_to_tiles_missing_metadata_raises(self):
        dataset = fo.Dataset()
        sample = fo.Sample(filepath="image1.png")
        dataset.add_sample(sample)
        with self.assertRaises(ValueError):
            dataset.to_tiles(tile_size=(640, 640))

    @drop_datasets
    def test_to_tiles_other_fields_default_true(self):
        dataset = fo.Dataset()
        sample = fo.Sample(
            filepath="image1.png",
            metadata=fo.ImageMetadata(width=640, height=640),
            predictions=fo.Detections(
                detections=[
                    fo.Detection(label="dog", bounding_box=[0, 0, 0.5, 0.5])
                ]
            ),
        )
        dataset.add_sample(sample)
        view = dataset.to_tiles(tile_size=(640, 640))
        self.assertEqual(len(view), 1)
        tile = view.first()
        self.assertIn("predictions", tile.field_names)

    @drop_datasets
    def test_to_tiles_is_patches_view(self):
        dataset = fo.Dataset()
        sample = fo.Sample(
            filepath="image1.png",
            metadata=fo.ImageMetadata(width=640, height=640),
        )
        dataset.add_sample(sample)
        view = dataset.to_tiles(tile_size=(640, 640))
        self.assertIsInstance(view, fop.PatchesView)

    @drop_datasets
    def test_to_tiles_multiple_samples(self):
        """Multiple samples with different sizes."""
        dataset = fo.Dataset()
        dataset.add_samples(
            [
                fo.Sample(
                    filepath="small.png",
                    metadata=fo.ImageMetadata(width=320, height=320),
                    ground_truth=fo.Detections(
                        detections=[
                            fo.Detection(
                                label="a", bounding_box=[0, 0, 0.5, 0.5]
                            ),
                        ]
                    ),
                ),
                fo.Sample(
                    filepath="large.png",
                    metadata=fo.ImageMetadata(width=1280, height=1280),
                    ground_truth=fo.Detections(
                        detections=[
                            fo.Detection(
                                label="b",
                                bounding_box=[0.5, 0.5, 0.3, 0.3],
                            ),
                        ]
                    ),
                ),
            ]
        )

        view = dataset.to_tiles(tile_size=(640, 640))

        # small.png: 320<640, so 1 tile. large.png: 2x2=4 tiles. Total=5
        self.assertEqual(len(view), 5)

        # Every tile should have ground_truth accessible
        for tile in view:
            self.assertIn("ground_truth", tile.field_names)

        # Check sample_id references back to source
        sample_ids = set(view.values("sample_id"))
        source_ids = set(dataset.values("id"))
        self.assertEqual(sample_ids, source_ids)

    @drop_datasets
    def test_to_tiles_with_other_fields_false(self):
        """other_fields=False should exclude label fields."""
        dataset = fo.Dataset()
        sample = fo.Sample(
            filepath="image.png",
            metadata=fo.ImageMetadata(width=640, height=640),
            ground_truth=fo.Detections(
                detections=[fo.Detection(label="x", bounding_box=[0, 0, 1, 1])]
            ),
        )
        dataset.add_sample(sample)

        view = dataset.to_tiles(tile_size=(640, 640), other_fields=False)
        tile = view.first()
        self.assertNotIn("ground_truth", tile.field_names)

    @drop_datasets
    def test_to_tiles_min_coverage(self):
        """min_coverage filters small edge tiles."""
        dataset = fo.Dataset()
        dataset.add_sample(
            fo.Sample(
                filepath="image.png",
                metadata=fo.ImageMetadata(width=700, height=700),
            )
        )

        view_all = dataset.to_tiles(tile_size=(640, 640), min_coverage=0.0)
        view_filtered = dataset.to_tiles(
            tile_size=(640, 640), min_coverage=0.5
        )

        # 700/640 => 2 cols, 2 rows => 4 tiles total
        self.assertEqual(len(view_all), 4)
        # Edge tiles are 60/640 = 0.09 of tile width, coverage < 0.5
        # Only the top-left tile (640x640 on 700x700) has full coverage
        self.assertLess(len(view_filtered), len(view_all))

    @drop_datasets
    def test_tile_field_cleaned_up(self):
        """The temporary tile_regions field should be removed after view creation."""
        dataset = fo.Dataset()
        dataset.add_sample(
            fo.Sample(
                filepath="image.png",
                metadata=fo.ImageMetadata(width=640, height=640),
            )
        )

        view = dataset.to_tiles(tile_size=(640, 640))
        # Force view materialization
        _ = len(view)

        # Temporary field should not linger on source dataset
        self.assertNotIn("tile_regions", dataset.get_field_schema())


if __name__ == "__main__":
    unittest.main()

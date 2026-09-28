"""
Preview tiling operator unit tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

import fiftyone as fo
import fiftyone.core.stages as fosg
import fiftyone.core.tiles as fot
from fiftyone import ViewField as F
from fiftyone.operators.executor import ExecutionContext, Executor

from decorators import drop_datasets
from plugins.tiles import PreviewTiling, _parse_params


def _ctx(dataset, view=None, **params):
    request_params = {"dataset_name": dataset.name, "params": params}
    if view is not None:
        request_params["view"] = view._serialize()

    return ExecutionContext(
        operator_uri="@vittorio-prodomo/tiles/preview_tiling",
        request_params=request_params,
        executor=Executor(),
    )


def _inputs(ctx):
    return PreviewTiling().resolve_input(ctx).type.properties


def _dataset(*sizes):
    dataset = fo.Dataset()
    dataset.add_samples(
        [
            fo.Sample(
                filepath="/tmp/preview%d.jpg" % i,
                metadata=fo.ImageMetadata(width=w, height=h),
                weather="rain" if i % 2 else "sun",
            )
            for i, (w, h) in enumerate(sizes)
        ]
    )
    return dataset


def _set_view(ctx):
    requests = [
        r for r in ctx.executor._requests if r.operator_uri == "set_view"
    ]
    return fo.DatasetView._build(ctx.dataset, requests[-1].params["view"])


class ParseParamsTests(unittest.TestCase):
    def test_pixels_and_percent(self):
        self.assertEqual(
            _parse_params(
                {"tile_width": 640, "tile_height": 480, "overlap": 64}
            ),
            ((640, 480), 64, 0),
        )
        self.assertEqual(
            _parse_params(
                {
                    "tile_width": 640,
                    "tile_height": 640,
                    "overlap": 25,
                    "overlap_unit": "%",
                    "min_label_coverage": 50,
                }
            ),
            ((640, 640), 0.25, 0.5),
        )

    def test_invalid_values(self):
        for params in (
            {"tile_width": None, "tile_height": 640},
            {"tile_width": 640, "tile_height": 640, "overlap": 640},
            {"tile_width": 640, "tile_height": 640, "overlap": 0.5},
            {
                "tile_width": 640,
                "tile_height": 640,
                "overlap": 100,
                "overlap_unit": "%",
            },
            {"tile_width": 640, "tile_height": 640, "min_label_coverage": 150},
        ):
            with self.assertRaises(ValueError, msg=str(params)):
                _parse_params(params)


class PreviewTilingTests(unittest.TestCase):
    @drop_datasets
    def test_placement_is_only_for_images(self):
        dataset = _dataset((1000, 500))
        self.assertIsNotNone(PreviewTiling().resolve_placement(_ctx(dataset)))

        videos = fo.Dataset()
        videos.media_type = "video"
        self.assertIsNone(PreviewTiling().resolve_placement(_ctx(videos)))

    @drop_datasets
    def test_form_defaults(self):
        dataset = _dataset((1000, 500))
        inputs = _inputs(_ctx(dataset))
        self.assertEqual(inputs["tile_width"].default, 640)
        self.assertEqual(inputs["overlap_unit"].default, "px")

        # an existing tiling is pre-filled, fractions as percentages
        tiles = dataset.to_tiles(
            (512, 256), overlap=0.25, min_label_coverage=0.5
        )
        inputs = _inputs(_ctx(dataset, view=tiles))
        self.assertEqual(inputs["tile_width"].default, 512)
        self.assertEqual(inputs["tile_height"].default, 256)
        self.assertEqual(inputs["overlap"].default, 25)
        self.assertEqual(inputs["overlap_unit"].default, "%")
        self.assertEqual(inputs["min_label_coverage"].default, 50)

    @drop_datasets
    def test_summary(self):
        # 1000x500 -> 2 clamped tiles; 700x700 -> 2 x 2 tiles
        dataset = _dataset((1000, 500), (700, 700), (700, 700))
        ctx = _ctx(dataset, tile_width=600, tile_height=600, overlap=100)
        summary = _inputs(ctx)["summary"].default
        self.assertIn("3 image(s) → 10 tiles", summary)
        self.assertIn("Most common grid: 2 × 2 tiles (2 image(s))", summary)
        self.assertIn("Overlap: 100 × 100 px", summary)
        self.assertIn("1 image(s) are smaller than a tile", summary)

        # the summary covers the images of the current view
        ctx = _ctx(
            dataset,
            view=dataset.match(F("metadata.width") == 700),
            tile_width=600,
            tile_height=600,
            overlap=100,
        )
        summary = _inputs(ctx)["summary"].default
        self.assertIn("2 image(s) → 8 tiles", summary)
        self.assertIn("Every image is cut into 2 × 2 tiles", summary)

    @drop_datasets
    def test_invalid_params_block_execution(self):
        dataset = _dataset((1000, 500))
        inputs = _inputs(
            _ctx(dataset, tile_width=600, tile_height=600, overlap=600)
        )
        self.assertTrue(inputs["invalid_params"].invalid)
        self.assertNotIn("summary", inputs)

    @drop_datasets
    def test_execute_sets_a_tiles_view(self):
        dataset = _dataset((1000, 500), (700, 700))
        ctx = _ctx(
            dataset,
            tile_width=600,
            tile_height=600,
            overlap=25,
            overlap_unit="%",
            min_label_coverage=10,
        )
        PreviewTiling().execute(ctx)

        view = _set_view(ctx)
        self.assertIsInstance(view, fot.TilesView)
        self.assertEqual(len(view), 2 + 4)

        stage = view._all_stages[-1]
        self.assertIsInstance(stage, fosg.ToTiles)
        self.assertEqual(stage.overlap, 0.25)
        self.assertEqual(stage.min_label_coverage, 0.1)

    @drop_datasets
    def test_execute_replaces_an_existing_tiling(self):
        dataset = _dataset((1000, 500), (700, 700), (1000, 500))
        view = (
            dataset.match(F("weather") == "sun").to_tiles((600, 600)).limit(1)
        )
        ctx = _ctx(dataset, view=view, tile_width=250, tile_height=250)
        PreviewTiling().execute(ctx)

        stages = _set_view(ctx)._all_stages
        self.assertEqual(
            [type(s) for s in stages],
            [fosg.Match, fosg.ToTiles, fosg.Limit],
        )
        self.assertEqual(tuple(stages[1].tile_size), (250, 250))

    @drop_datasets
    def test_missing_metadata(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "image.png")
        Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8)).save(path)

        dataset = _dataset((1000, 500))
        dataset.add_sample(fo.Sample(filepath=path))

        inputs = _inputs(_ctx(dataset, tile_width=200, tile_height=200))
        self.assertIn("compute_metadata", inputs)
        self.assertTrue(inputs["missing_metadata"].invalid)

        ctx = _ctx(
            dataset, tile_width=200, tile_height=200, compute_metadata=True
        )
        self.assertNotIn("missing_metadata", _inputs(ctx))

        PreviewTiling().execute(ctx)
        dataset.reload()
        self.assertEqual(
            dataset.match(F("filepath") == path).first().metadata.width, 400
        )
        # 1000x500 -> 5 x 3; 400x300 -> 2 x 2
        self.assertEqual(len(_set_view(ctx)), 15 + 4)


if __name__ == "__main__":
    fo.config.show_progress_bars = False
    unittest.main(verbosity=2)

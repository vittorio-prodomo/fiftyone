"""
Tiles plugin operator unit tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import shutil
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
from plugins.tiles import ExportTiles, PreviewTiling, _parse_params


def _ctx(dataset, view=None, **params):
    request_params = {"dataset_name": dataset.name, "params": params}
    if view is not None:
        request_params["view"] = view._serialize()

    return ExecutionContext(
        operator_uri="@vittorio-prodomo/tiles/preview_tiling",
        request_params=request_params,
        executor=Executor(),
    )


def _inputs(ctx, operator=None):
    operator = operator or PreviewTiling()
    return operator.resolve_input(ctx).type.properties


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
    def test_summary_of_uncontained_objects(self):
        # 1000x500 -> 600x500 tiles at x=0 and x=400
        dataset = _dataset((1000, 500))
        sample = dataset.first()
        sample["gt"] = fo.Detections(
            detections=[
                fo.Detection(bounding_box=[0.05, 0.1, 0.1, 0.1]),
                fo.Detection(bounding_box=[0.35, 0.1, 0.3, 0.1]),
                fo.Detection(bounding_box=[0.1, 0.1, 0.7, 0.1]),
            ]
        )
        sample["pred"] = fo.Detections(
            detections=[fo.Detection(bounding_box=[0.05, 0.1, 0.1, 0.1])]
        )
        sample["empty"] = fo.Detections()
        sample.save()

        ctx = _ctx(dataset, tile_width=600, tile_height=600, overlap=100)
        summary = _inputs(ctx)["summary"].default
        self.assertIn("Objects that no tile fully contains", summary)
        self.assertIn(
            "- `gt`: 2 of 3 (66.7%) — 1 larger than a tile, 1 crossing "
            "tile borders",
            summary,
        )
        self.assertIn("- `pred`: none of 1", summary)
        self.assertNotIn("empty", summary)

        # images without metadata are left out, as in the rest of the summary
        dataset.add_sample(
            fo.Sample(
                filepath="/tmp/nometa.jpg",
                gt=fo.Detections(
                    detections=[
                        fo.Detection(bounding_box=[0.35, 0.1, 0.3, 0.1])
                    ]
                ),
            )
        )
        summary = _inputs(_ctx(dataset, tile_width=600, tile_height=600))[
            "summary"
        ].default
        self.assertIn("- `gt`: 2 of 3", summary)

    @drop_datasets
    def test_no_uncontained_objects_section_without_labels(self):
        dataset = _dataset((1000, 500))
        ctx = _ctx(dataset, tile_width=600, tile_height=600)
        summary = _inputs(ctx)["summary"].default
        self.assertNotIn("Objects", summary)

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


class ExportTilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _dataset(self):
        """Two 1000x500 images cut into 600x500 tiles at x=0 and x=400."""
        dataset = fo.Dataset()
        for i, labels in enumerate(
            (
                [fo.Detection(label="cat", bounding_box=[0.1, 0.1, 0.1, 0.2])],
                [fo.Detection(label="dog", bounding_box=[0.8, 0.1, 0.1, 0.2])],
            )
        ):
            path = os.path.join(self.tmp, "src", "img%d.png" % i)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Image.fromarray(np.zeros((500, 1000, 3), dtype=np.uint8)).save(
                path
            )
            dataset.add_sample(
                fo.Sample(
                    filepath=path,
                    gt=fo.Detections(detections=labels),
                    polys=fo.Polylines(),
                    weather=fo.Classification(label="sunny"),
                )
            )

        dataset.compute_metadata()
        return dataset

    def _ctx(self, dataset, view, **params):
        request_params = {
            "dataset_name": dataset.name,
            "view": view._serialize(),
            "params": params,
        }
        return ExecutionContext(
            operator_uri="@vittorio-prodomo/tiles/export_tiles",
            request_params=request_params,
            executor=Executor(),
        )

    @drop_datasets
    def test_placement_is_only_for_tiles_views(self):
        dataset = self._dataset()
        operator = ExportTiles()
        self.assertIsNone(
            operator.resolve_placement(self._ctx(dataset, dataset.view()))
        )
        tiles = dataset.to_tiles((600, 600))
        self.assertIsNotNone(
            operator.resolve_placement(self._ctx(dataset, tiles))
        )

        inputs = _inputs(self._ctx(dataset, dataset.view()), ExportTiles())
        self.assertTrue(inputs["not_tiles"].invalid)

    @drop_datasets
    def test_form(self):
        dataset = self._dataset()
        tiles = dataset.to_tiles((600, 600))

        def inputs(**params):
            return _inputs(self._ctx(dataset, tiles, **params), ExportTiles())

        # YOLOv5 by default, with the fields it can export
        props = inputs()
        self.assertEqual(props["format"].default, "yolov5")
        self.assertEqual(props["label_field"].type.values, ["gt", "polys"])
        # YOLOv5 decides which label files exist for negative tiles
        self.assertEqual(
            props["label_files_empty_tiles"].type.values,
            ["keep", "keep_without_labels", "skip"],
        )
        self.assertEqual(props["label_files_empty_tiles"].default, "keep")
        self.assertEqual(
            props["label_files_empty_tiles"].view.label,
            "Negative Tiles (i.e., without labels)",
        )
        self.assertNotIn("empty_tiles", props)
        self.assertEqual(props["split"].default, "train")
        self.assertNotIn("join_polygon_parts", props)

        summary = props["summary"].default
        self.assertIn("**4 tile(s)** of 2 image(s)", summary)
        self.assertIn("2 tile(s) have no labels in `gt`", summary)
        self.assertIn("2 classes", summary)
        self.assertIn("0 `cat`, 1 `dog`", summary)

        # polylines can join their parts
        self.assertIn("join_polygon_parts", inputs(label_field="polys"))

        # other formats offer other fields, no split, and only to keep or
        # skip negative tiles
        props = inputs(format="classification")
        self.assertEqual(props["label_field"].type.values, ["weather"])
        self.assertNotIn("split", props)
        self.assertEqual(props["empty_tiles"].type.values, ["keep", "skip"])

        props = inputs(format="coco")
        self.assertEqual(props["label_field"].type.values, ["gt", "polys"])
        self.assertEqual(props["empty_tiles"].type.values, ["keep", "skip"])
        self.assertEqual(
            props["empty_tiles"].view.label,
            "Negative Tiles (i.e., without labels)",
        )
        self.assertNotIn("label_files_empty_tiles", props)

        # other formats with a label file per image also decide which ones
        # exist
        for fmt in ("yolov4", "kitti"):
            props = inputs(format=fmt)
            self.assertEqual(
                props["label_files_empty_tiles"].type.values,
                ["keep", "keep_without_labels", "skip"],
                msg=fmt,
            )
            self.assertNotIn("empty_tiles", props)
            self.assertNotIn("split", props)

        props = inputs(format="images")
        self.assertNotIn("label_field", props)
        self.assertNotIn("empty_tiles", props)

        # no field to export
        tiles = dataset.exclude_fields("weather").to_tiles((600, 600))
        props = _inputs(
            self._ctx(dataset, tiles, format="classification"), ExportTiles()
        )
        self.assertTrue(props["no_fields"].invalid)

    @drop_datasets
    def test_execute(self):
        dataset = self._dataset()
        tiles = dataset.to_tiles((600, 600))
        export_dir = os.path.join(self.tmp, "yolo")

        # the tiles of img1 only have the dog, which keeps index 1
        view = tiles.match(F("filepath").ends_with("img1.png"))
        ctx = self._ctx(
            dataset,
            view,
            format="yolov5",
            label_field="gt",
            export_dir={"absolute_path": export_dir},
            label_files_empty_tiles="skip",
            split="val",
        )
        result = ExportTiles().execute(ctx)

        self.assertEqual(result["num_tiles"], 1)
        images = os.listdir(os.path.join(export_dir, "images", "val"))
        self.assertEqual(images, ["img1_tile_0_1.png"])
        with open(
            os.path.join(export_dir, "labels", "val", "img1_tile_0_1.txt")
        ) as f:
            self.assertTrue(f.read().startswith("1 "))

        with open(os.path.join(export_dir, "dataset.yaml")) as f:
            yaml = f.read()

        self.assertIn("0: cat", yaml)
        self.assertIn("1: dog", yaml)

        # every tile, without labels
        export_dir = os.path.join(self.tmp, "images")
        ctx = self._ctx(
            dataset,
            tiles,
            format="images",
            export_dir={"absolute_path": export_dir},
        )
        self.assertEqual(ExportTiles().execute(ctx)["num_tiles"], 4)
        self.assertEqual(len(os.listdir(export_dir)), 4)

        # other formats keep or skip negative tiles, and a choice made for
        # YOLOv5 does not carry over to them
        for params, num_tiles in (
            ({"empty_tiles": "skip"}, 2),
            ({"label_files_empty_tiles": "skip"}, 4),
        ):
            export_dir = os.path.join(self.tmp, "detection%d" % num_tiles)
            ctx = self._ctx(
                dataset,
                tiles,
                format="fiftyone_detection",
                label_field="gt",
                export_dir={"absolute_path": export_dir},
                **params,
            )
            result = ExportTiles().execute(ctx)
            self.assertEqual(result["num_tiles"], num_tiles, msg=params)

        # negative tiles without label files, in other formats with a label
        # file per image
        for empty_tiles, num_label_files in (
            ("keep", 4),
            ("keep_without_labels", 2),
        ):
            export_dir = os.path.join(self.tmp, "kitti_" + empty_tiles)
            ctx = self._ctx(
                dataset,
                tiles,
                format="kitti",
                label_field="gt",
                export_dir={"absolute_path": export_dir},
                label_files_empty_tiles=empty_tiles,
            )
            self.assertEqual(ExportTiles().execute(ctx)["num_tiles"], 4)
            data = os.listdir(os.path.join(export_dir, "data"))
            labels = os.listdir(os.path.join(export_dir, "labels"))
            self.assertEqual(len(data), 4)
            self.assertEqual(len(labels), num_label_files, msg=empty_tiles)

        export_dir = os.path.join(self.tmp, "yolov4")
        ctx = self._ctx(
            dataset,
            tiles,
            format="yolov4",
            label_field="gt",
            export_dir={"absolute_path": export_dir},
            label_files_empty_tiles="keep_without_labels",
        )
        self.assertEqual(ExportTiles().execute(ctx)["num_tiles"], 4)
        files = os.listdir(os.path.join(export_dir, "data"))
        self.assertEqual(len([f for f in files if f.endswith(".png")]), 4)
        self.assertEqual(len([f for f in files if f.endswith(".txt")]), 2)
        with open(os.path.join(export_dir, "images.txt")) as f:
            self.assertEqual(len(f.read().split()), 4)

        # an export directory is required
        ctx = self._ctx(dataset, tiles, format="images")
        with self.assertRaises(ValueError):
            ExportTiles().execute(ctx)

    def _split_dataset(self):
        """Four 1000x500 images, cut into 600x500 tiles at x=0 and x=400:
        two tagged ``train``, one ``val``, and one with no split.
        """
        dataset = self._dataset()
        copies = []
        for sample in dataset:
            filepath = sample.filepath.replace(".png", "_copy.png")
            shutil.copy(sample.filepath, filepath)
            copies.append(fo.Sample(filepath=filepath, gt=sample.gt))

        dataset.add_samples(copies)
        dataset.compute_metadata()
        ids = dataset.values("id")
        dataset.select(ids[:2]).tag_samples("train")
        dataset.select(ids[2]).tag_samples("val")
        dataset.select(ids[3]).tag_samples("other")
        dataset.add_sample_field("split", fo.StringField)
        dataset.set_values("split", ["a", "b", "a", None])
        return dataset

    @drop_datasets
    def test_split_form(self):
        dataset = self._split_dataset()
        tiles = dataset.to_tiles((600, 600))

        def inputs(**params):
            return _inputs(self._ctx(dataset, tiles, **params), ExportTiles())

        # no splits by default, and YOLOv5's single split
        props = inputs()
        self.assertEqual(props["split_by"].default, "none")
        self.assertEqual(
            props["split_by"].type.values, ["none", "tags", "field", "random"]
        )
        self.assertIn("split", props)
        self.assertNotIn("split_tags", props)

        # the split tags that the images have are proposed
        props = inputs(split_by="tags")
        self.assertNotIn("split", props)
        self.assertEqual(props["split_tags"].default, ["train", "val"])
        self.assertEqual(
            [c.value for c in props["split_tags"].view.choices],
            ["other", "train", "val"],
        )
        summary = props["summary"].default
        self.assertIn("`train`: 4 tile(s) of 2 image(s)", summary)
        self.assertIn("`val`: 2 tile(s) of 1 image(s)", summary)
        self.assertIn("1 image(s) are in no split", summary)

        # an image in two splits is invalid
        props = inputs(split_by="tags", split_tags=["train", "other", "val"])
        self.assertNotIn("invalid_splits", props)
        dataset.select(dataset.first().id).tag_samples("val")
        props = inputs(split_by="tags", split_tags=["train", "val"])
        self.assertTrue(props["invalid_splits"].invalid)
        self.assertNotIn("summary", props)

        # string fields of the images, whose values are the splits
        props = inputs(split_by="field")
        self.assertEqual(props["split_field"].type.values, ["split"])
        summary = props["summary"].default
        self.assertIn("`a`: 4 tile(s) of 2 image(s)", summary)
        self.assertIn("`b`: 2 tile(s) of 1 image(s)", summary)

        # random splits leave out the splits of 0
        props = inputs(split_by="random")
        self.assertEqual(
            {
                k: p.default
                for k, p in props["split_fracs"].type.properties.items()
            },
            {"train": 0.8, "val": 0.2, "test": 0.0},
        )
        self.assertEqual(props["split_seed"].default, 51)
        self.assertFalse(props["tag_splits"].default)
        summary = props["summary"].default
        self.assertIn("`train`: 6 tile(s) of 3 image(s)", summary)
        self.assertIn("`val`: 2 tile(s) of 1 image(s)", summary)
        self.assertNotIn("`test`", summary)

        props = inputs(
            split_by="random",
            split_fracs={"train": 0, "val": 0, "test": 0},
        )
        self.assertTrue(props["invalid_splits"].invalid)

    @drop_datasets
    def test_split_execute(self):
        dataset = self._split_dataset()
        tiles = dataset.to_tiles((600, 600))

        # YOLOv5 splits by tags, in one dataset.yaml
        export_dir = os.path.join(self.tmp, "yolo")
        ctx = self._ctx(
            dataset,
            tiles,
            format="yolov5",
            label_field="gt",
            export_dir={"absolute_path": export_dir},
            split_by="tags",
            split_tags=["train", "val"],
        )
        result = ExportTiles().execute(ctx)
        self.assertEqual(result["num_tiles"], 6)
        self.assertEqual(result["splits"], "train: 4, val: 2")
        for split, num_images in (("train", 4), ("val", 2)):
            images_dir = os.path.join(export_dir, "images", split)
            self.assertEqual(len(os.listdir(images_dir)), num_images)

        with open(os.path.join(export_dir, "dataset.yaml")) as f:
            yaml = f.read()

        self.assertIn("train: ./images/train/", yaml)
        self.assertIn("val: ./images/val/", yaml)

        # other formats, in a directory per split
        export_dir = os.path.join(self.tmp, "coco")
        ctx = self._ctx(
            dataset,
            tiles,
            format="coco",
            label_field="gt",
            export_dir={"absolute_path": export_dir},
            split_by="field",
            split_field="split",
        )
        result = ExportTiles().execute(ctx)
        self.assertEqual(result["splits"], "a: 4, b: 2")
        self.assertEqual(sorted(os.listdir(export_dir)), ["a", "b"])

        # a random split, saved as tags of the images, which replace their
        # tags of the random splits
        export_dir = os.path.join(self.tmp, "random")
        ctx = self._ctx(
            dataset,
            tiles,
            format="images",
            export_dir={"absolute_path": export_dir},
            split_by="random",
            split_fracs={"train": 0.5, "val": 0.5, "test": 0},
            split_seed=3,
            tag_splits=True,
        )
        result = ExportTiles().execute(ctx)
        self.assertEqual(result["num_tiles"], 8)
        self.assertEqual(result["splits"], "train: 4, val: 4")
        self.assertEqual(dataset.count_sample_tags()["train"], 2)
        self.assertEqual(dataset.count_sample_tags()["val"], 2)
        self.assertEqual(
            len(dataset.match_tags(["train", "val"], all=True)), 0
        )
        self.assertEqual(dataset.count_sample_tags()["other"], 1)
        for split in ("train", "val"):
            stems = {
                f.rsplit("_tile_", 1)[0]
                for f in os.listdir(os.path.join(export_dir, split))
            }
            expected = {
                os.path.splitext(os.path.basename(p))[0]
                for p in dataset.match_tags(split).values("filepath")
            }
            self.assertEqual(stems, expected)


if __name__ == "__main__":
    fo.config.show_progress_bars = False
    unittest.main(verbosity=2)

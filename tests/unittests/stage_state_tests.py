"""
Conversion stage state round-trip unit tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import unittest

import fiftyone as fo
from fiftyone import ViewField as F

from decorators import drop_datasets


class ConversionStageStateTests(unittest.TestCase):
    """Conversion stages record the stages before them in a ``_state``
    snapshot. Rebuilding a view from JSON, as the App's requests do, must not
    turn the expressions in that snapshot into ViewExpressions, whose ``==``
    builds an expression instead of comparing.
    """

    def _dataset(self):
        dataset = fo.Dataset()
        dataset.add_samples(
            [
                fo.Sample(
                    filepath="/tmp/state%d.jpg" % i,
                    metadata=fo.ImageMetadata(width=1000, height=500),
                    weather="sun" if i else "rain",
                    gt=fo.Detections(
                        detections=[
                            fo.Detection(
                                label="a", bounding_box=[0.1, 0.1, 0.2, 0.2]
                            )
                        ]
                    ),
                )
                for i in range(3)
            ]
        )
        return dataset

    def _check_round_trip(self, dataset, view):
        rebuilt = fo.DatasetView._build(dataset, view._serialize())

        self.assertEqual(len(rebuilt), len(view))

        # the unchanged state is recognized, so the generated dataset is
        # reused rather than rebuilt
        self.assertEqual(
            rebuilt._patches_dataset.name, view._patches_dataset.name
        )

    @drop_datasets
    def test_patches_after_an_expression(self):
        dataset = self._dataset()
        view = dataset.match(F("weather") == "sun").to_patches("gt")
        self._check_round_trip(dataset, view)

    @drop_datasets
    def test_tiles_after_an_expression(self):
        dataset = self._dataset()
        view = dataset.match(F("weather") == "sun").to_tiles((600, 600))
        self._check_round_trip(dataset, view)


if __name__ == "__main__":
    fo.config.show_progress_bars = False
    unittest.main(verbosity=2)

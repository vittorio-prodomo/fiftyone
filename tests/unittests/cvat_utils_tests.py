"""
Unit tests for ``fiftyone.utils.cvat`` helpers.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import unittest

import fiftyone as fo  # noqa: F401  bootstrap modules to avoid circular imports
import fiftyone.utils.cvat as fouc
from fiftyone.utils.cvat import _BasenameLookup


class TestBasenameLookup(unittest.TestCase):
    """Tests for :class:`_BasenameLookup`."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_empty(self):
        lookup = _BasenameLookup([])
        self.assertEqual(len(lookup), 0)
        self.assertFalse(lookup)

    def test_single_file(self):
        lookup = _BasenameLookup(["/data/images/cat.jpg"])
        self.assertEqual(len(lookup), 1)
        self.assertTrue(lookup)

    # ------------------------------------------------------------------
    # .get() — unique basenames
    # ------------------------------------------------------------------

    def test_get_bare_basename(self):
        lookup = _BasenameLookup(["/data/images/cat.jpg"])
        self.assertEqual(lookup.get("cat.jpg"), "/data/images/cat.jpg")

    def test_get_with_subdir(self):
        lookup = _BasenameLookup(["/data/images/train/cat.jpg"])
        self.assertEqual(
            lookup.get("train/cat.jpg"), "/data/images/train/cat.jpg"
        )

    def test_get_missing_returns_default(self):
        lookup = _BasenameLookup(["/data/images/cat.jpg"])
        self.assertIsNone(lookup.get("dog.jpg"))
        self.assertEqual(lookup.get("dog.jpg", "fallback"), "fallback")

    # ------------------------------------------------------------------
    # .get() — disambiguation (multiple files with same basename)
    # ------------------------------------------------------------------

    def test_get_disambiguates_by_subdir(self):
        paths = [
            "/data/images/train/cat.jpg",
            "/data/images/val/cat.jpg",
        ]
        lookup = _BasenameLookup(paths)
        self.assertEqual(
            lookup.get("train/cat.jpg"), "/data/images/train/cat.jpg"
        )
        self.assertEqual(lookup.get("val/cat.jpg"), "/data/images/val/cat.jpg")

    def test_get_ambiguous_bare_basename_returns_default(self):
        paths = [
            "/data/images/train/cat.jpg",
            "/data/images/val/cat.jpg",
        ]
        lookup = _BasenameLookup(paths)
        # bare basename is ambiguous — should return default
        self.assertIsNone(lookup.get("cat.jpg"))

    def test_get_ambiguous_suffix_returns_default(self):
        paths = [
            "/data/project/images/train/cat.jpg",
            "/data/archive/images/train/cat.jpg",
        ]
        lookup = _BasenameLookup(paths)
        # "images/train/cat.jpg" matches both — ambiguous
        self.assertIsNone(lookup.get("images/train/cat.jpg"))

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def test_getitem_found(self):
        lookup = _BasenameLookup(["/data/images/cat.jpg"])
        self.assertEqual(lookup["cat.jpg"], "/data/images/cat.jpg")

    def test_getitem_missing_raises_keyerror(self):
        lookup = _BasenameLookup(["/data/images/cat.jpg"])
        with self.assertRaises(KeyError):
            _ = lookup["dog.jpg"]

    def test_getitem_ambiguous_raises_keyerror(self):
        paths = [
            "/data/images/train/cat.jpg",
            "/data/images/val/cat.jpg",
        ]
        lookup = _BasenameLookup(paths)
        with self.assertRaises(KeyError):
            _ = lookup["cat.jpg"]

    # ------------------------------------------------------------------
    # __contains__ (``in`` operator)
    # ------------------------------------------------------------------

    def test_contains_found(self):
        lookup = _BasenameLookup(["/data/images/cat.jpg"])
        self.assertIn("cat.jpg", lookup)

    def test_contains_missing(self):
        lookup = _BasenameLookup(["/data/images/cat.jpg"])
        self.assertNotIn("dog.jpg", lookup)

    def test_contains_ambiguous(self):
        paths = [
            "/data/images/train/cat.jpg",
            "/data/images/val/cat.jpg",
        ]
        lookup = _BasenameLookup(paths)
        self.assertNotIn("cat.jpg", lookup)
        self.assertIn("train/cat.jpg", lookup)

    # ------------------------------------------------------------------
    # __len__, __bool__
    # ------------------------------------------------------------------

    def test_len(self):
        paths = ["/data/a.jpg", "/data/b.jpg", "/data/c.jpg"]
        lookup = _BasenameLookup(paths)
        self.assertEqual(len(lookup), 3)

    def test_bool_empty(self):
        self.assertFalse(_BasenameLookup([]))

    def test_bool_nonempty(self):
        self.assertTrue(_BasenameLookup(["/data/a.jpg"]))

    # ------------------------------------------------------------------
    # Iteration, keys, values, items
    # ------------------------------------------------------------------

    def test_iter(self):
        paths = ["/data/a.jpg", "/data/b.jpg"]
        lookup = _BasenameLookup(paths)
        self.assertEqual(list(lookup), paths)

    def test_keys(self):
        paths = ["/data/a.jpg", "/data/b.jpg"]
        lookup = _BasenameLookup(paths)
        self.assertEqual(list(lookup.keys()), paths)

    def test_values(self):
        paths = ["/data/a.jpg", "/data/b.jpg"]
        lookup = _BasenameLookup(paths)
        self.assertEqual(list(lookup.values()), paths)

    def test_items(self):
        paths = ["/data/a.jpg", "/data/b.jpg"]
        lookup = _BasenameLookup(paths)
        self.assertEqual(list(lookup.items()), [(p, p) for p in paths])

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    @unittest.skipUnless(os.sep == "\\", "backslash separator is Windows-only")
    def test_backslash_separator(self):
        """On Windows, backslash separators in queries are normalized."""
        lookup = _BasenameLookup(["/data/images/train/cat.jpg"])
        self.assertEqual(
            lookup.get("train\\cat.jpg"), "/data/images/train/cat.jpg"
        )

    def test_deeply_nested_path(self):
        path = "/data/a/b/c/d/e/f/img.jpg"
        lookup = _BasenameLookup([path])
        self.assertEqual(lookup.get("d/e/f/img.jpg"), path)
        self.assertEqual(lookup.get("img.jpg"), path)

    def test_iterated_keys_round_trip(self):
        """Iterated keys must resolve back through get/contains/getitem."""
        paths = [
            "/data/images/train/cat.jpg",
            "/data/images/val/dog.jpg",
        ]
        lookup = _BasenameLookup(paths)
        for key in lookup:
            self.assertIn(key, lookup)
            self.assertEqual(lookup[key], key)
            self.assertEqual(lookup.get(key), key)

    def test_iterated_keys_round_trip_ambiguous_basenames(self):
        """Round-trip must work even when basenames collide."""
        paths = [
            "/data/images/train/cat.jpg",
            "/data/images/val/cat.jpg",
        ]
        lookup = _BasenameLookup(paths)
        for key in lookup:
            self.assertIn(key, lookup)
            self.assertEqual(lookup[key], key)

    def test_duplicate_filepaths_deduplicated(self):
        paths = ["/data/a.jpg", "/data/b.jpg", "/data/a.jpg"]
        lookup = _BasenameLookup(paths)
        self.assertEqual(len(lookup), 2)
        self.assertEqual(list(lookup), ["/data/a.jpg", "/data/b.jpg"])


class _FakeResponse(object):
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class _FakeCVATAPI(object):
    """Returns a canned ``tasks/{id}/data/meta`` response so that
    :func:`fiftyone.utils.cvat._parse_task_metadata` runs without a server.
    """

    def __init__(self, meta):
        self._meta = meta

    def task_data_meta_url(self, task_id):
        return "http://cvat.test/api/tasks/%s/data/meta" % task_id

    def get(self, url):
        return _FakeResponse(self._meta)


class TestParseTaskMetadataFrameFilters(unittest.TestCase):
    """Job frame ranges and CVAT's deleted frames both exclude frames."""

    def _parse(self, deleted_frames=None, frame_ranges=None):
        meta = {
            "start_frame": 0,
            "stop_frame": 5,
            "chunk_size": 10,
            "frames": [{"name": "img%d.jpg" % i} for i in range(6)],
        }
        if deleted_frames is not None:
            meta["deleted_frames"] = deleted_frames

        data_map = {"img%d.jpg" % i: "/data/img%d.jpg" % i for i in range(6)}
        cvat_id_map = fouc._parse_task_metadata(
            _FakeCVATAPI(meta),
            1,
            data_map,
            [],
            [],
            [],
            frame_ranges=frame_ranges,
        )
        return sorted(cvat_id_map.values())

    def test_no_filters_keeps_every_frame(self):
        self.assertEqual(self._parse(), [0, 1, 2, 3, 4, 5])

    def test_frame_ranges_are_inclusive(self):
        self.assertEqual(self._parse(frame_ranges=[(1, 2), (4, 4)]), [1, 2, 4])

    def test_deleted_frames_are_skipped(self):
        self.assertEqual(self._parse(deleted_frames=[0, 3]), [1, 2, 4, 5])

    def test_both_filters_combine(self):
        self.assertEqual(
            self._parse(deleted_frames=[2], frame_ranges=[(1, 3)]), [1, 3]
        )


class TestCVATResultsJobIdsFilter(unittest.TestCase):
    """The imported job IDs survive a save/load of the annotation run, so
    that ``load_annotations()`` only downloads those jobs.
    """

    def _results(self, job_ids_filter):
        return fouc.CVATAnnotationResults(
            None,
            fouc.CVATBackendConfig("cvat", {}),
            "anno",
            {},
            {},
            [],
            [7],
            {7: [70, 71, 72]},
            {7: {}},
            {"ground_truth": [7]},
            job_ids_filter=job_ids_filter,
        )

    def _round_trip(self, results):
        d = results.serialize()
        return fouc.CVATAnnotationResults._from_dict(
            d, None, results.config, "anno"
        )

    def test_filter_round_trips(self):
        results = self._round_trip(self._results({71, 72}))
        self.assertEqual(sorted(results.job_ids_filter), [71, 72])

    def test_no_filter_round_trips_as_none(self):
        results = self._round_trip(self._results(None))
        self.assertIsNone(results.job_ids_filter)

    def test_runs_saved_before_the_filter_existed_still_load(self):
        d = self._results(None).serialize()
        d.pop("job_ids_filter", None)
        results = fouc.CVATAnnotationResults._from_dict(
            d, None, fouc.CVATBackendConfig("cvat", {}), "anno"
        )
        self.assertIsNone(results.job_ids_filter)


if __name__ == "__main__":
    fo.config.show_progress_bars = False
    unittest.main(verbosity=2)

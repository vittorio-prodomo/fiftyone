# Tiled Patches View Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Add a `to_tiles()` API that generates a grid of tile bounding boxes
over images and creates a patches view showing all annotations within each
tile, plus a frontend slider to control zoom padding.

**Architecture:** Generate synthetic `Detection` bounding boxes as tiles, store
in a temporary `_tile_regions` field, delegate to existing
`to_patches("_tile_regions", other_fields=True)`. Frontend already renders all
labels on patches correctly — only need to wire `zoomPad` to a UI slider.

**Tech Stack:** Python (FiftyOne core), TypeScript/React (FiftyOne App), Recoil
(state management)

---

### Task 1: Tile Grid Generation Helper

**Files:**

-   Create: `fiftyone/core/tiles.py`
-   Test: `tests/unittests/tiles_tests.py`

**Step 1: Write the failing test**

Create `tests/unittests/tiles_tests.py`:

```python
"""
FiftyOne tiles-related unit tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import unittest

from fiftyone.core.tiles import compute_tile_detections


class ComputeTileDetectionsTests(unittest.TestCase):
    def test_exact_fit_no_overlap(self):
        """640x640 tiles on a 1280x1280 image => 4 tiles, no overlap."""
        dets = compute_tile_detections(
            img_w=1280, img_h=1280, tile_w=640, tile_h=640, overlap=0
        )
        self.assertEqual(len(dets), 4)
        # top-left tile
        self._assert_bbox(dets[0], 0.0, 0.0, 0.5, 0.5)
        # top-right tile
        self._assert_bbox(dets[1], 0.5, 0.0, 0.5, 0.5)
        # bottom-left tile
        self._assert_bbox(dets[2], 0.0, 0.5, 0.5, 0.5)
        # bottom-right tile
        self._assert_bbox(dets[3], 0.5, 0.5, 0.5, 0.5)

    def test_non_divisible_produces_edge_tiles(self):
        """1000x1000 image with 640x640 tiles => 4 tiles, edge tiles are smaller."""
        dets = compute_tile_detections(
            img_w=1000, img_h=1000, tile_w=640, tile_h=640, overlap=0
        )
        self.assertEqual(len(dets), 4)
        # last column tile should be clipped to image boundary
        self.assertAlmostEqual(
            dets[1].bounding_box[0] + dets[1].bounding_box[2], 1.0, places=6
        )

    def test_overlap_pixels(self):
        """Overlap specified in pixels (>= 1)."""
        dets = compute_tile_detections(
            img_w=1280, img_h=1280, tile_w=640, tile_h=640, overlap=64
        )
        # With 64px overlap on 1280px, stride=576, positions: 0, 576, 1152
        # But 1152+640>1280, so last tile clipped. 3 cols x 3 rows = 9? Let's check:
        # stride_x = 640 - 64 = 576. starts: 0, 576, 1152. 1152+640=1792>1280 => clip to 1280
        # 3 x 3 = 9
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
        # Some edge tiles should be filtered out
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


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd tests/unittests && python -m pytest tiles_tests.py -v` Expected: FAIL
with `ModuleNotFoundError: No module named 'fiftyone.core.tiles'`

**Step 3: Write minimal implementation**

Create `fiftyone/core/tiles.py`:

```python
"""
Tile generation utilities.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import math

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
    y_px = 0.0
    while y_px < img_h:
        col = 0
        x_px = 0.0
        while x_px < img_w:
            # Clip tile to image boundaries
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

            x_px += stride_x
            col += 1

        y_px += stride_y
        row += 1

    return detections
```

**Step 4: Run test to verify it passes**

Run: `cd tests/unittests && python -m pytest tiles_tests.py -v` Expected: All 7
tests PASS

**Step 5: Commit**

```bash
git add fiftyone/core/tiles.py tests/unittests/tiles_tests.py
git commit -m "feat: add tile grid generation utility"
```

---

### Task 2: `ToTiles` ViewStage and `to_tiles()` Method

**Files:**

-   Modify: `fiftyone/core/stages.py:8241` (insert `ToTiles` class after
    `ToPatches`)
-   Modify: `fiftyone/core/stages.py:9327` (add `ToTiles` to `_STAGES` list)
-   Modify: `fiftyone/core/collections.py:8172` (add `to_tiles()` method after
    `to_patches()`)
-   Modify: `fiftyone/__public__.py:249` (export `ToTiles`)
-   Test: `tests/unittests/tiles_tests.py` (add integration test class)

**Step 1: Write the failing test**

Add to `tests/unittests/tiles_tests.py`:

```python
import fiftyone as fo
import fiftyone.core.patches as fop

from decorators import drop_datasets


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
                        label="cat",
                        bounding_box=[0.1, 0.1, 0.3, 0.3],
                    ),
                ]
            ),
        )
        dataset.add_sample(sample)

        view = dataset.to_tiles(tile_size=(640, 640))

        # 2x2 grid = 4 tiles
        self.assertEqual(len(view), 4)

        # Each tile should have the ground_truth field
        tile = view.first()
        self.assertIn("ground_truth", tile.field_names)

    @drop_datasets
    def test_to_tiles_with_overlap(self):
        """Tiling with pixel overlap."""
        dataset = fo.Dataset()
        sample = fo.Sample(
            filepath="image1.png",
            metadata=fo.ImageMetadata(width=1280, height=1280),
        )
        dataset.add_sample(sample)

        view = dataset.to_tiles(tile_size=(640, 640), overlap=64)

        # stride=576, starts: 0, 576, 1152 => 3 cols x 3 rows = 9
        self.assertEqual(len(view), 9)

    @drop_datasets
    def test_to_tiles_missing_metadata_raises(self):
        """Should raise ValueError if metadata is missing."""
        dataset = fo.Dataset()
        sample = fo.Sample(filepath="image1.png")
        dataset.add_sample(sample)

        with self.assertRaises(ValueError):
            dataset.to_tiles(tile_size=(640, 640))

    @drop_datasets
    def test_to_tiles_other_fields_default_true(self):
        """other_fields defaults to True, bringing all label fields."""
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

        # Single tile (exact fit), predictions should be present
        self.assertEqual(len(view), 1)
        tile = view.first()
        self.assertIn("predictions", tile.field_names)

    @drop_datasets
    def test_to_tiles_is_patches_view(self):
        """Result should be a PatchesView."""
        dataset = fo.Dataset()
        sample = fo.Sample(
            filepath="image1.png",
            metadata=fo.ImageMetadata(width=640, height=640),
        )
        dataset.add_sample(sample)

        view = dataset.to_tiles(tile_size=(640, 640))
        self.assertIsInstance(view, fop.PatchesView)
```

**Step 2: Run test to verify it fails**

Run: `cd tests/unittests && python -m pytest tiles_tests.py::ToTilesTests -v`
Expected: FAIL with `AttributeError: ... has no attribute 'to_tiles'`

**Step 3: Write the implementation**

**3a. Add `ToTiles` ViewStage to `fiftyone/core/stages.py`**

Insert after line 8241 (after `ToPatches` class, before `ToEvaluationPatches`):

```python
class ToTiles(ViewStage):
    """Creates a view that contains one sample per tile in a grid over the
    images in a collection.

    A ``sample_id`` field will be added that records the sample ID from which
    each tile was taken.

    All other label fields are included by default so annotations are visible
    within each tile.

    Samples must have their ``metadata`` field populated.

    Examples::

        import fiftyone as fo
        import fiftyone.zoo as foz

        dataset = foz.load_zoo_dataset("quickstart")
        dataset.compute_metadata()

        session = fo.launch_app(dataset)

        # Create a view containing 640x640 tiles
        stage = fo.ToTiles(tile_size=(640, 640), overlap=64)
        view = dataset.add_stage(stage)
        print(view)

        session.view = view

    Args:
        tile_size: a ``(width, height)`` tuple specifying the tile dimensions
            in pixels
        overlap (0): overlap between adjacent tiles. Values >= 1 are treated
            as pixels, values < 1 as fractions of tile size
        other_fields (True): controls whether fields other than the tile
            regions and the default sample fields are included. Can be any of
            the following:

            -   a field or list of fields to include
            -   ``True`` to include all other fields
            -   ``None``/``False`` to include no other fields
        min_coverage (0.0): minimum fraction of tile area that must be within
            the image boundary for edge tiles
        config (None): an optional dict of keyword arguments for
            :meth:`fiftyone.core.patches.make_patches_dataset`
        **kwargs: optional keyword arguments for
            :meth:`fiftyone.core.patches.make_patches_dataset`
    """

    _TILE_FIELD = "_tile_regions"

    def __init__(
        self,
        tile_size,
        overlap=0,
        other_fields=True,
        min_coverage=0.0,
        config=None,
        _state=None,
        **kwargs,
    ):
        if kwargs:
            if config is None:
                config = kwargs
            else:
                config.update(kwargs)

        self._tile_size = tile_size
        self._overlap = overlap
        self._other_fields = other_fields
        self._min_coverage = min_coverage
        self._config = config
        self._state = _state

    @property
    def has_view(self):
        return True

    @property
    def tile_size(self):
        """The ``(width, height)`` tile dimensions in pixels."""
        return self._tile_size

    @property
    def overlap(self):
        """The overlap between adjacent tiles."""
        return self._overlap

    @property
    def other_fields(self):
        """Controls which other fields are included."""
        return self._other_fields

    @property
    def min_coverage(self):
        """The minimum tile coverage fraction."""
        return self._min_coverage

    @property
    def config(self):
        """Parameters specifying how to perform the conversion."""
        return self._config

    def load_view(self, sample_collection, saved_view=False, reload=False):
        from fiftyone.core.tiles import compute_tile_detections

        state = {
            "dataset_id": str(sample_collection._root_dataset._doc.id),
            "stages": sample_collection.view()._serialize(include_uuids=False),
            "tile_size": list(self._tile_size),
            "overlap": self._overlap,
            "other_fields": self._other_fields,
            "min_coverage": self._min_coverage,
            "config": self._config,
        }

        last_state = deepcopy(self._state)
        if last_state is not None:
            name = last_state.pop("name", None)
        else:
            name = None

        try:
            last_dataset = fod.load_dataset(name, reload=True)
        except:
            last_dataset = None

        if (
            reload
            or last_dataset is None
            or (state != last_state and not saved_view)
        ):
            # Generate tile detections for each sample
            tile_w, tile_h = self._tile_size
            _add_tile_regions(
                sample_collection,
                self._TILE_FIELD,
                tile_w,
                tile_h,
                self._overlap,
                self._min_coverage,
            )

            kwargs = deepcopy(self._config) or {}
            if "other_fields" not in kwargs:
                kwargs["other_fields"] = self._other_fields

            if reload and last_dataset is not None:
                kwargs["include_indexes"] = last_dataset

            patches_dataset = fop.make_patches_dataset(
                sample_collection,
                self._TILE_FIELD,
                _generated=True,
                **kwargs,
            )

            if name is not None and (saved_view or state == last_state):
                if last_dataset is not None:
                    last_dataset._delete()
                patches_dataset.name = name
        else:
            patches_dataset = last_dataset

        state["name"] = patches_dataset.name
        self._state = state

        return fop.PatchesView(sample_collection, self, patches_dataset)

    def _kwargs(self):
        return [
            ["tile_size", self._tile_size],
            ["overlap", self._overlap],
            ["other_fields", self._other_fields],
            ["min_coverage", self._min_coverage],
            ["config", self._config],
            ["_state", self._state],
        ]

    @classmethod
    def _params(cls):
        return [
            {
                "name": "tile_size",
                "type": "list<int>",
                "placeholder": "(width, height)",
            },
            {
                "name": "overlap",
                "type": "int|float",
                "default": "0",
                "placeholder": "overlap (default=0)",
            },
            {
                "name": "other_fields",
                "type": "NoneType|bool|list<str>",
                "default": "True",
            },
            {
                "name": "min_coverage",
                "type": "float",
                "default": "0.0",
            },
            {
                "name": "config",
                "type": "NoneType|json",
                "default": "None",
            },
            {"name": "_state", "type": "NoneType|json", "default": "None"},
        ]


def _add_tile_regions(
    sample_collection, field, tile_w, tile_h, overlap, min_coverage
):
    """Adds tile Detection objects to each sample in the collection.

    Requires ``metadata`` to be populated on all samples.
    """
    from fiftyone.core.tiles import compute_tile_detections

    fol = fou.lazy_import("fiftyone.core.labels")

    # Ensure metadata exists
    missing = sample_collection.exists("metadata", False)
    if len(missing) > 0:
        raise ValueError(
            "Found %d sample(s) without metadata. You must run "
            "`dataset.compute_metadata()` before calling `to_tiles()`"
            % len(missing)
        )

    # Add the tile field if it doesn't exist
    if not sample_collection._root_dataset.has_sample_field(field):
        sample_collection._root_dataset.add_sample_field(
            field,
            fof.EmbeddedDocumentField,
            embedded_doc_type=fol.Detections,
        )

    # Generate tiles per sample
    for sample in sample_collection.iter_samples(autosave=True, progress=True):
        img_w = sample.metadata.width
        img_h = sample.metadata.height
        dets = compute_tile_detections(
            img_w, img_h, tile_w, tile_h, overlap, min_coverage
        )
        sample[field] = fol.Detections(detections=dets)
```

Note: `_add_tile_regions` uses `fou` which is already imported in `stages.py`.
However, `fof` and `fol` are needed — add these imports at the function level
or at the top of the file. Check existing imports in `stages.py` to follow the
pattern (they use `fof = fou.lazy_import(...)` at the top).

**3b. Register `ToTiles` in `_STAGES` list at `stages.py:9327`**

Add `ToTiles,` after `ToPatches,`:

```python
ToPatches,
ToTiles,
ToEvaluationPatches,
```

**3c. Add `to_tiles()` to `fiftyone/core/collections.py`**

Insert after `to_patches()` (after line 8172):

```python
@view_stage
def to_tiles(
    self, tile_size, overlap=0, other_fields=True, min_coverage=0.0, **kwargs
):
    """Creates a view that contains one sample per tile in a grid over the
    images in the collection.

    All other label fields are included by default so that annotations are
    visible within each tile.

    Samples must have their ``metadata`` field populated. You can ensure
    this by running :meth:`compute_metadata`.

    Examples::

        import fiftyone as fo
        import fiftyone.zoo as foz

        dataset = foz.load_zoo_dataset("quickstart")
        dataset.compute_metadata()

        session = fo.launch_app(dataset)

        # Create a view containing 640x640 tiles with 64px overlap
        view = dataset.to_tiles(tile_size=(640, 640), overlap=64)
        print(view)

        session.view = view

    Args:
        tile_size: a ``(width, height)`` tuple specifying the tile
            dimensions in pixels
        overlap (0): overlap between adjacent tiles. Values >= 1 are
            treated as pixels, values < 1 as fractions of tile size
        other_fields (True): controls whether fields other than the tile
            regions and the default sample fields are included. Can be
            any of the following:

            -   a field or list of fields to include
            -   ``True`` to include all other fields
            -   ``None``/``False`` to include no other fields
        min_coverage (0.0): minimum fraction of tile area that must be
            within the image boundary for edge tiles

    Returns:
        a :class:`fiftyone.core.patches.PatchesView`
    """
    return self._add_view_stage(
        fos.ToTiles(
            tile_size,
            overlap=overlap,
            other_fields=other_fields,
            min_coverage=min_coverage,
            **kwargs,
        )
    )
```

**3d. Export `ToTiles` from `fiftyone/__public__.py`**

At line 249, add `ToTiles,` after `ToPatches,`:

```python
ToPatches,
ToTiles,
ToEvaluationPatches,
```

**Step 4: Run test to verify it passes**

Run: `cd tests/unittests && python -m pytest tiles_tests.py -v` Expected: All
tests PASS

**Step 5: Commit**

```bash
git add fiftyone/core/stages.py fiftyone/core/collections.py fiftyone/__public__.py tests/unittests/tiles_tests.py
git commit -m "feat: add ToTiles view stage and to_tiles() method"
```

---

### Task 3: Frontend — `zoomPad` Slider in Grid Options

**Files:**

-   Modify: `app/packages/state/src/recoil/atoms.ts:73` (add `zoomPad` atom)
-   Modify: `app/packages/state/src/recoil/looker.ts:107` (wire `zoomPad` into
    options)
-   Modify:
    `app/packages/core/src/components/Actions/Options/Options.tsx:66-84,437`
    (add ZoomPad control)

**Step 1: Add `zoomPad` Recoil atom**

In `app/packages/state/src/recoil/atoms.ts`, after line 73 (`cropToContent`
atom):

```typescript
export const zoomPad = atomFamily<number, boolean>({
    key: "zoomPad",
    default: 0.2,
});
```

**Step 2: Wire `zoomPad` into looker options**

In `app/packages/state/src/recoil/looker.ts`, at line 107 (after the `zoom`
line), add:

```typescript
        zoomPad: get(atoms.zoomPad(modal)),
```

So lines 107-108 become:

```typescript
        zoom: get(viewAtoms.isPatchesView) && get(atoms.cropToContent(modal)),
        zoomPad: get(atoms.zoomPad(modal)),
```

**Step 3: Add ZoomPad control component**

In `app/packages/core/src/components/Actions/Options/Options.tsx`, add a
`ZoomPad` component after the `Patches` component (after line 84):

```tsx
const ZoomPad = ({ modal }: { modal: boolean }) => {
    const isPatches = useRecoilValue(fos.isPatchesView);
    const [zoomPad, setZoomPad] = useRecoilState(fos.zoomPad(modal));

    if (!isPatches) {
        return null;
    }

    return (
        <>
            <PopoutSectionTitle>Zoom padding</PopoutSectionTitle>
            <Slider
                value={zoomPad}
                onChange={(e, value) => setZoomPad(value as number)}
                min={0}
                max={0.5}
                step={0.05}
            />
        </>
    );
};
```

Note: Check which slider component is available. If `Slider` is not imported,
use the pattern from other controls in Options.tsx. The exact component may be
from `@mui/material` or `@fiftyone/components`. Follow existing patterns.

**Step 4: Add ZoomPad to Options render**

At line 437, add `<ZoomPad modal={!!modal} />` after the `<Patches>` line:

```tsx
{
    mode === fos.EXPLORE && <Patches modal={!!modal} />;
}
{
    mode === fos.EXPLORE && <ZoomPad modal={!!modal} />;
}
```

**Step 5: Export the atom from state package**

Ensure `zoomPad` is exported from the state package's public API. Check
`app/packages/state/src/recoil/index.ts` or equivalent barrel file and add the
export if needed.

**Step 6: Verify**

Run: `cd app && yarn build` (or the project's build command) Expected: No
TypeScript errors

**Step 7: Commit**

```bash
git add app/packages/state/src/recoil/atoms.ts app/packages/state/src/recoil/looker.ts app/packages/core/src/components/Actions/Options/Options.tsx
git commit -m "feat: add zoomPad slider to grid options for patches views"
```

---

### Task 4: Integration Smoke Test

**Files:**

-   Test: `tests/unittests/tiles_tests.py` (add end-to-end-style test)

**Step 1: Add an integration test**

Add to `ToTilesTests` class in `tests/unittests/tiles_tests.py`:

```python
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
                        fo.Detection(label="a", bounding_box=[0, 0, 0.5, 0.5]),
                    ]
                ),
            ),
            fo.Sample(
                filepath="large.png",
                metadata=fo.ImageMetadata(width=1280, height=1280),
                ground_truth=fo.Detections(
                    detections=[
                        fo.Detection(
                            label="b", bounding_box=[0.5, 0.5, 0.3, 0.3]
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
    view_filtered = dataset.to_tiles(tile_size=(640, 640), min_coverage=0.5)

    # 700/640 => 2 cols, 2 rows => 4 tiles total
    self.assertEqual(len(view_all), 4)
    # Edge tiles are 60/640 = 0.09 of tile width, coverage < 0.5
    # Only the top-left tile (640x640 on 700x700) has full coverage
    self.assertLess(len(view_filtered), len(view_all))
```

**Step 2: Run all tests**

Run: `cd tests/unittests && python -m pytest tiles_tests.py -v` Expected: All
tests PASS

**Step 3: Commit**

```bash
git add tests/unittests/tiles_tests.py
git commit -m "test: add integration tests for to_tiles"
```

---

### Task 5: Cleanup — Remove Temporary `_tile_regions` Field

**Files:**

-   Modify: `fiftyone/core/stages.py` (add cleanup in `ToTiles.load_view`)

**Step 1: Add cleanup logic**

The `_tile_regions` field is written to the source dataset. It should be
cleaned up when the tiles view is no longer needed. Add cleanup in
`ToTiles.load_view` — after creating the patches dataset, delete the temporary
field from the source:

In the `load_view` method, after
`patches_dataset = fop.make_patches_dataset(...)` and before setting
`state["name"]`, add:

```python
# Clean up temporary tile field from source dataset
if sample_collection._root_dataset.has_sample_field(self._TILE_FIELD):
    sample_collection._root_dataset.delete_sample_field(self._TILE_FIELD)
```

**Step 2: Add test for cleanup**

Add to `ToTilesTests`:

```python
@drop_datasets
def test_tile_field_cleaned_up(self):
    """The temporary _tile_regions field should be removed after view creation."""
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
    self.assertNotIn("_tile_regions", dataset.get_field_schema())
```

**Step 3: Run test**

Run:
`cd tests/unittests && python -m pytest tiles_tests.py::ToTilesTests::test_tile_field_cleaned_up -v`
Expected: PASS

**Step 4: Commit**

```bash
git add fiftyone/core/stages.py tests/unittests/tiles_tests.py
git commit -m "fix: clean up temporary _tile_regions field after tile view creation"
```

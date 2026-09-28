# Follow-ups after the September 2026 upstream sync

Open work left after syncing the three forks (`fiftyone`, `fiftyone-brain`,
`eta`) with upstream in September 2026. Details and reproductions live in the
linked PR threads.

## Repository state

- All three forks follow upstream's trunk-based layout: `main` is the default
  branch, and `develop` has been deleted (upstream renamed `develop` to `main`
  in August 2026).
- Sync upstream by merging `upstream/main` into a branch and opening a PR into
  `main`. Merge with "Create a merge commit" (never squash or rebase), so later
  syncs keep recognizing upstream's history.
- PRs into `main` run upstream's full CI (`build`, `test`, `test-windows`,
  `e2e`). A PR that changes `setup.py`'s `VERSION` needs the `version-bump`
  label.
- The dependency pins the forks need locally (`graphql-core<3.3`, and overrides
  for fiftyone's `fiftyone-brain` and `voxel51-eta` pins) are in the cloud
  environment's setup script.

## Upstream contributions

Bugs found in upstream code during the sync. The forks carry upstream's code
unchanged for all of them except the first.

### 1. Conversion stages fail to rebuild after an expression stage (fiftyone)

- **Status**: fixed in this fork (2ba1eb6b95), still broken on upstream `main`
- **Bug**: since upstream 817e18dd88 ("decode view expression envelopes when
  loading stages"), `ViewStage._from_dict()` also decodes the `_state` snapshot
  of conversion stages. `load_view()` then compares it with
  `state != last_state`, which calls `ViewExpression.__eq__` and raises
  `TypeError`. So any view like `dataset.match(F("x") == 1).to_patches("gt")`
  fails to rebuild from JSON, which is how the App and operators receive views.
- **Fix**: leave `_state` undecoded in `ViewStage._from_dict()`; tests are in
  `tests/unittests/stage_state_tests.py`
- **Next step**: open an upstream PR with the fix and the tests

### 2. 3D annotation imports an unexported helper (fiftyone App)

- **Status**: confirmed on upstream `main`, not fixed
- **Bug**: upstream b16366178c removed `export * from "./utils"` from
  `app/packages/core/src/components/Modal/Sidebar/Annotate/index.tsx`, but
  `coerceStringBooleans` and its imports in `looker-3d`
  (`use-3d-annotation.ts`, `use3dAnnotationEventHandlers.ts`) came back later.
  The import resolves to `undefined`, so editing a 3D label throws at runtime
  (the production build does not catch it).
- **Fix**: restore the barrel export, or drop the helper and unwrap the two
  call sites, as b16366178c intended
- **Thread**:
  https://github.com/vittorio-prodomo/fiftyone/pull/1#discussion_r4120852334

### 3. `add_ids()` skips existing IDs only when warning (fiftyone-brain)

- **Status**: reproduced
- **Bug**: in `fiftyone/brain/internal/core/utils.py`, `add_ids()` only removes
  existing IDs inside its `elif warn_existing:` branch. With `overwrite=False`,
  existing points are overwritten when `warn_existing` is `False` and kept when
  it is `True`. Affects the visualization `add_samples(skip_existing=False)`
  and `SklearnSimilarityIndex.add_to_index(overwrite=False)`.
- **Fix**: run the `np.delete` whenever `overwrite` is false and
  `allow_existing` is true; only the warning is conditional
- **Thread**:
  https://github.com/vittorio-prodomo/fiftyone-brain/pull/1#discussion_r4120801368

### 4. Patch-level `add_samples()` with an array fails (fiftyone-brain)

- **Status**: reproduced
- **Bug**: in `fiftyone/brain/visualization.py`, `_array_to_id_dict()` keys
  patch embeddings by label ID, but `fbu.get_embeddings()` looks dicts up by
  sample ID. The update fails with
  `embeddings have dimension 0 but the fitted reducer expects dimension N`.
- **Fix**: group the rows into one `(num_labels, dim)` array per sample ID, and
  have `_filter_known()` drop known label IDs within each group
- **Thread**:
  https://github.com/vittorio-prodomo/fiftyone-brain/pull/1#discussion_r4120800828

### 5. Custom visualization backends lost their `fit()` hook (fiftyone-brain)

- **Status**: confirmed
- **Bug**: upstream renamed `Visualization.fit()` to `fit_reducer()` (cf50b46),
  so custom backends registered in `BrainConfig.visualization_methods` that
  only define `fit()` now raise `NotImplementedError`
- **Fix**: have the base `fit_reducer()` fall back to `self.fit()` and return
  `(points, None)`
- **Thread**:
  https://github.com/vittorio-prodomo/fiftyone-brain/pull/1#discussion_r4120799585

### 6. eta's Sphinx docs no longer build (eta)

- **Status**: tested fix
- **Bug**: upstream's move to `pyproject.toml` dropped `m2r` from the `dev`
  extra, while `sphinx/source/conf.py` still loads it. Re-adding it does not
  help: `m2r` crashes on current docutils, and `m2r2` renders the README as an
  empty page.
- **Fix**: add `myst-parser` to the `dev` extra, replace `"m2r"` with
  `"myst_parser"` in `conf.py` (dropping `m2r_parse_relative_links`), and
  include the README with `:parser: myst_parser.sphinx_`
- **Thread**:
  https://github.com/vittorio-prodomo/eta/pull/1#discussion_r4120744121

### 7. `write_json()` output with custom date fields cannot be loaded (fiftyone)

- **Status**: fixed in this fork (branch `fix/json-datetime-roundtrip`), still
  broken on upstream `main`
- **Bug**: `write_json()` / `to_dict()` serialize dates as extended JSON
  (`{"$date": "..."}`), but `deserialize_value()` in
  `fiftyone/core/odm/utils.py` only turns `$oid` and `$binary` back into Python
  objects. A custom `DateTimeField` or `DateField` (top-level, inside a list,
  or as a label attribute) stays a dict, and `Dataset.from_json()` /
  `from_dict()` fails with
  `ValueError: Invalid value for field 'capture_time'. Reason: Datetime fields must have datetime values`.
  Built-in `created_at` / `last_modified_at` are unaffected because they are
  reset on insert.
- **Fix**: decode `$date` dicts with `json_util` in `deserialize_value()`; test
  `DatasetSerializationTests.test_serialize_dataset_dates` in
  `tests/unittests/dataset_tests.py`
- **Next step**: open an upstream PR with the fix and the test

## Tiles follow-ups

The tiles view (`fiftyone/core/tiles.py`, `ToTiles`, and the "Preview tiling"
action in `plugins/tiles/`) is a lightweight preview of how a dataset looks
when tiled: a view, not a data transformation. It uses the "shift" edge mode,
clamps tiles that are larger than the image, keeps every label that touches a
tile (optional `min_label_coverage`), and keeps tags on the tiles only.

1.  **Labels never fully inside any tile**: add to the Preview tiling summary
    the number of objects that no tile fully contains. This shows when the tile
    size or overlap is too small for the objects, before applying the tiling.
2.  **Live re-zoom in the modal**: changing the Zoom padding slider does not
    re-zoom the image already open in the modal; it applies to the next sample.
    Re-run the zoom on change (`zoomPad` feeds
    `app/packages/looker/src/zoom.ts`).
3.  **Export a tiled training set**: turn a tiles view into real data by
    writing the tile crops and their labels, clipped to the tile and
    re-normalized to the crop. This is the step from previewing a tiling to
    training on it.
4.  **Tags**: tags set on tiles stay local to the tiles view. Decide whether
    they should sync back to the source samples or labels once there is a use
    case.

# Modal Image Download Button — Design

## Problem

FiftyOne's modal viewer has no way to download the full-resolution image being
viewed. Users can see their images but cannot easily save individual samples to
disk. This is a basic expectation from a dataset visualization tool.

## Solution

Add a download button in two locations in the modal UI, backed by a new server
endpoint that forces browser download behavior.

## Architecture

### Backend: `/media/download` endpoint

New Starlette endpoint registered alongside the existing `/media` route.

```
GET /media/download?filepath={encoded_path}
```

Reuses the same file-serving logic as `/media` (stat check, MIME type guessing,
streaming in 8192-byte chunks) but adds:

-   `Content-Disposition: attachment; filename="<basename>"` header to force
    download
-   Filename extracted via `os.path.basename(filepath)`

**File:** `fiftyone/server/routes/media.py` (add `MediaDownload` class)
**File:** `fiftyone/server/routes/__init__.py` (register route)

### Frontend: Download button — two locations

**Location 1: Modal top action bar**

New `Download` action component in
`app/packages/core/src/components/Modal/Actions/`. Icon button alongside
fullscreen, tag, options, etc.

**Location 2: Looker bottom controls overlay**

Download action in `app/packages/looker/src/elements/common/actions.ts` — the
hover bar with zoom, JSON toggle, crop-to-content controls.

### Download mechanism

Both buttons:

1. Read the current sample's media URL from existing state
2. Replace `/media?filepath=...` with `/media/download?filepath=...`
3. Create a temporary `<a>` element with `href` and `download` attribute, click
   it programmatically

### User feedback

-   On click: button disabled, icon replaced with spinner
-   After ~2 seconds: spinner removed, button re-enabled
-   Prevents double-clicks; browser's native download manager handles progress

## What we don't build

-   **Batch download** — single-sample action only
-   **Format conversion** — original file as-is
-   **Custom filenames** — uses original filename from filepath
-   **Auth/authz** — follows existing `/media` security model
-   **Download progress bar** — browser native download UI handles this

# Modal Image Download Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Add a download button (in both the modal action bar and the looker
bottom controls) that downloads the current sample's media file via a new
server endpoint with `Content-Disposition: attachment`.

**Architecture:** New `MediaDownload` Starlette endpoint reuses existing file
serving logic but forces download. Frontend adds a download button in two
locations: the React-based modal action bar and the vanilla-JS looker controls
bar. Both construct a `/media/download?filepath=...` URL from the current
sample's media URL. A 2-second debounce spinner prevents double-clicks.

**Tech Stack:** Python (Starlette), TypeScript/React (Modal actions),
TypeScript vanilla DOM (Looker controls)

---

### Task 1: Backend — `/media/download` endpoint

**Files:**

-   Modify: `fiftyone/server/routes/media.py` (add `MediaDownload` class after
    `Media` class, ~line 156)
-   Modify: `fiftyone/server/routes/__init__.py` (register route and import)

**Step 1: Write the endpoint**

Add to `fiftyone/server/routes/media.py` after the `Media` class:

```python
class MediaDownload(HTTPEndpoint):
    async def get(
        self, request: Request
    ) -> t.Union[FileResponse, StreamingResponse]:
        path = request.query_params["filepath"]

        try:
            await anyio.to_thread.run_sync(os.stat, path)
        except FileNotFoundError:
            return Response(content="Not found", status_code=404)

        filename = os.path.basename(path)
        content_type = guess_type(path)[0] or "application/octet-stream"

        response = FileResponse(
            path,
            media_type=content_type,
            filename=filename,
        )
        response.headers["Content-Disposition"] = (
            'attachment; filename="%s"' % filename
        )
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers[
            "Access-Control-Allow-Headers"
        ] = "Content-Type, Authorization"

        return response
```

Note: Starlette's `FileResponse` already sets `Content-Length` and handles
streaming. The explicit `Content-Disposition: attachment` header guarantees
download behavior across all browsers and origins.

**Step 2: Register the route**

In `fiftyone/server/routes/__init__.py`:

Add to the import at line 22:

```python
from .media import Media, MediaDownload
```

Add to the routes list, after the `/media` line (~line 46):

```python
("/media/download", MediaDownload),
```

**Step 3: Verify manually**

Start FiftyOne server and test:

```bash
curl -I "http://localhost:5151/media/download?filepath=/path/to/any/image.jpg"
```

Expected: Response headers include
`Content-Disposition: attachment; filename="image.jpg"`

**Step 4: Commit**

```bash
git add fiftyone/server/routes/media.py fiftyone/server/routes/__init__.py
git commit -m "feat: add /media/download endpoint with Content-Disposition attachment"
```

---

### Task 2: Frontend — Download button in modal action bar

**Files:**

-   Create: `app/packages/core/src/components/Modal/Actions/Download/index.tsx`
-   Modify: `app/packages/core/src/components/Modal/Actions/index.tsx` (import
    and render)

**Step 1: Create the Download component**

Create `app/packages/core/src/components/Modal/Actions/Download/index.tsx`:

```tsx
import { PillButton } from "@fiftyone/components";
import {
    getSampleSrc,
    modalSample,
    selectedMediaField,
} from "@fiftyone/state";
import { Download as DownloadIcon } from "@mui/icons-material";
import { CircularProgress } from "@mui/material";
import React, { useCallback, useState } from "react";
import { useRecoilValue } from "recoil";

const Download = () => {
    const [downloading, setDownloading] = useState(false);
    const { sample, urls } = useRecoilValue(modalSample);
    const mediaField = useRecoilValue(selectedMediaField(true));

    const handleDownload = useCallback(() => {
        if (downloading) return;

        const rawUrl =
            urls?.[mediaField] ?? urls?.filepath ?? sample?.filepath;
        if (!rawUrl) return;

        // getSampleSrc builds "/media?filepath=..." — swap to "/media/download"
        const mediaSrc = getSampleSrc(rawUrl as string);
        const downloadSrc = mediaSrc.replace("/media?", "/media/download?");

        const a = document.createElement("a");
        a.href = downloadSrc;
        a.download = "";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        setDownloading(true);
        setTimeout(() => setDownloading(false), 2000);
    }, [downloading, urls, mediaField, sample]);

    return (
        <PillButton
            icon={
                downloading ? (
                    <CircularProgress size={16} thickness={6} />
                ) : (
                    <DownloadIcon />
                )
            }
            open={false}
            highlight={false}
            onClick={handleDownload}
            tooltipPlacement="bottom"
            title="Download media"
            data-cy="action-download"
            style={{ opacity: downloading ? 0.5 : undefined }}
        />
    );
};

export default Download;
```

**Step 2: Add Download to the modal action bar**

In `app/packages/core/src/components/Modal/Actions/index.tsx`:

Add import after the ToggleFullscreen import (~line 18):

```tsx
import Download from "./Download";
```

Add `<Download />` in the JSX, before `<ToggleFullscreen />` (~line 112):

```tsx
        <Download />
        <ToggleFullscreen />
```

**Step 3: Verify**

Open the FiftyOne app, click a sample to open modal, verify the download icon
appears in the top action bar. Click it — should trigger browser download.

**Step 4: Commit**

```bash
git add app/packages/core/src/components/Modal/Actions/Download/index.tsx \
       app/packages/core/src/components/Modal/Actions/index.tsx
git commit -m "feat: add download button to modal action bar"
```

---

### Task 3: Frontend — Download button in looker bottom controls

**Files:**

-   Modify: `app/packages/looker/src/icons/index.ts` (add download SVG icon)
-   Modify: `app/packages/looker/src/elements/common/controls.ts` (add
    `DownloadButtonElement` class)
-   Modify: `app/packages/looker/src/elements/common/actions.ts` (add
    `download` Control)
-   Modify: `app/packages/looker/src/elements/index.ts` (add
    `DownloadButtonElement` to image, frame, video, imavid element trees)

**Step 1: Add download SVG icon**

In `app/packages/looker/src/icons/index.ts`, add after the `json` export (~line
57):

```typescript
export const download = HTMLToDom(
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="var(--fo-palette-text-secondary)"><path d="M5,20H19V18H5M19,9H15V3H9V9H5L12,16L19,9Z" /></svg>'
);
```

This is the Material Design "download" icon (mdi-download) matching the style
of other icons in this file (24x24, `--fo-palette-text-secondary` fill).

**Step 2: Add download Control action**

In `app/packages/looker/src/elements/common/actions.ts`, add after the
`selectSample` control (~line 370):

```typescript
export const downloadMedia: Control = {
    title: "Download",
    shortcut: "d",
    detail: "Download the source media file",
    action: (update, dispatchEvent) => {
        dispatchEvent("download");
    },
};
```

Add `downloadMedia` to the `COMMON` map (~line 386):

```typescript
export const COMMON = {
    escape,
    rotateNext,
    rotatePrevious,
    help,
    zoomIn,
    zoomOut,
    cropToContent,
    resetZoom,
    controlsToggle,
    settings,
    json,
    wheel,
    toggleOverlays,
    selectSample,
    downloadMedia,
};
```

**Step 3: Add DownloadButtonElement**

In `app/packages/looker/src/elements/common/controls.ts`:

Add `download as downloadIcon` to the icons import (~line 6):

```typescript
import {
    crop,
    download as downloadIcon,
    help as helpIcon,
    json as jsonIcon,
    minus,
    options,
    overlaysHidden,
    overlaysVisible,
    plus,
} from "../../icons";
```

Add `downloadMedia` to the actions import (~line 18):

```typescript
import {
    cropToContent,
    downloadMedia,
    help,
    json,
    settings,
    toggleOverlays,
    zoomIn,
    zoomOut,
} from "./actions";
```

Add the class after `JSONButtonElement` (~line 344):

```typescript
export class DownloadButtonElement<
    State extends BaseState
> extends BaseElement<State> {
    private spinning = false;
    private timeoutId: ReturnType<typeof setTimeout> | null = null;

    getEvents(): Events<State> {
        return {
            click: ({ event, update, dispatchEvent }) => {
                event.stopPropagation();
                event.preventDefault();
                if (this.spinning) return;
                downloadMedia.action(update, dispatchEvent);
                this.setSpinning();
            },
        };
    }

    createHTMLElement() {
        const element = document.createElement("div");
        element.classList.add(lookerClickable);
        element.style.padding = "2px";
        element.style.display = "flex";
        element.title = `${downloadMedia.title} (${downloadMedia.shortcut})`;
        element.style.gridArea = "2 / 12 / 2 / 12";
        element.appendChild(downloadIcon.cloneNode(true));
        return element;
    }

    renderSelf() {
        return this.element;
    }

    private setSpinning() {
        this.spinning = true;
        this.element.style.opacity = "0.5";
        this.element.style.cursor = "wait";
        if (this.timeoutId) clearTimeout(this.timeoutId);
        this.timeoutId = setTimeout(() => {
            this.spinning = false;
            this.element.style.opacity = "1";
            this.element.style.cursor = "pointer";
        }, 2000);
    }
}
```

**Step 4: Add DownloadButtonElement to element trees**

In `app/packages/looker/src/elements/index.ts`, the download button needs to
appear in all looker types that have a controls bar.

For `getImageElements` (~line 117), `getFrameElements` (~line 68), and
`getVideoElements` (~line 168), add `{ node: common.DownloadButtonElement }`
inside the `ControlsElement` children, after `CropToContentButtonElement`:

```typescript
          { node: common.CropToContentButtonElement },
          { node: common.DownloadButtonElement },
          { node: common.ToggleOverlaysButtonElement },
```

For `getImaVidElements` (~line 237), same insertion:

```typescript
          { node: common.CropToContentButtonElement },
          { node: common.DownloadButtonElement },
          { node: common.ToggleOverlaysButtonElement },
```

**Step 5: Handle the "download" event in the React layer**

The `downloadMedia` action dispatches a `"download"` event. This needs to be
handled where looker events are consumed. Check
`app/packages/core/src/components/Modal/use-looker.ts` — this is where looker
events like `"options"`, `"panels"`, etc. are handled.

Add a handler for the `"download"` event that constructs the download URL and
triggers the browser download, same logic as the modal action bar button:

```typescript
looker.addEventListener("download", () => {
    const mediaSrc = looker.state.config.src;
    if (!mediaSrc) return;

    const downloadSrc = mediaSrc.replace("/media?", "/media/download?");
    const a = document.createElement("a");
    a.href = downloadSrc;
    a.download = "";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
});
```

**Step 6: Verify**

Open modal, hover over image to show controls bar, verify download icon
appears. Click it — spinner for 2s, triggers download.

**Step 7: Commit**

```bash
git add app/packages/looker/src/icons/index.ts \
       app/packages/looker/src/elements/common/actions.ts \
       app/packages/looker/src/elements/common/controls.ts \
       app/packages/looker/src/elements/index.ts \
       app/packages/core/src/components/Modal/use-looker.ts
git commit -m "feat: add download button to looker bottom controls bar"
```

---

### Task 4: Keyboard shortcut — "d" key triggers download

**Files:**

-   Already handled by Task 3 (the `downloadMedia` Control has `shortcut: "d"`)

The `downloadMedia` control registered in `COMMON` with `shortcut: "d"` means
pressing "d" while the looker is focused will dispatch the `"download"` event.
The event handler from Task 3 Step 5 picks it up.

**Step 1: Verify**

Open modal, press "d" — should trigger download.

**Step 2: Verify help panel**

Press "?" to open help — "Download" should appear in the shortcuts list with
"d" as the key.

No commit needed — this is covered by Task 3.

---

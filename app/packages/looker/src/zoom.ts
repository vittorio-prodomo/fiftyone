import {
  DYNAMIC_EMBEDDED_DOCUMENT_FIELD,
  Schema,
  Stage,
  getCls,
  getFieldInfo,
} from "@fiftyone/utilities";
import { MIN_PIXELS } from "./constants";
import { POINTS_FROM_FO } from "./overlays";
import { Overlay } from "./overlays/base";
import {
  BaseState,
  BoundingBox,
  Coordinates,
  Dimensions,
  FrameState,
  ImageState,
  VideoState,
} from "./state";
import { getContainingBox, mergeUpdates, snapBox } from "./util";

const PATCHES_STAGE_CLASSES = [
  "fiftyone.core.stages.ToPatches",
  "fiftyone.core.stages.ToEvaluationPatches",
];

const TILES_STAGE_CLASS = "fiftyone.core.stages.ToTiles";
const TILES_FIELD = "tile_regions";

/**
 * Extracts the primary patches field name from view stages.
 * Returns null if the view is not a patches view.
 */
export const getPatchesField = (view: Stage[]): string | null => {
  for (const stage of view) {
    if (PATCHES_STAGE_CLASSES.includes(stage._cls)) {
      const fieldKwarg = stage.kwargs.find(([key]) => key === "field");
      return (fieldKwarg?.[1] as unknown as string) ?? null;
    }
    if (stage._cls === TILES_STAGE_CLASS) {
      return TILES_FIELD;
    }
  }
  return null;
};

/**
 * Filters overlays to only the primary patches field when in a patches view.
 * Returns all overlays unchanged if not in a patches view.
 */
export const filterOverlaysForZoom = <State extends BaseState>(
  view: Stage[],
  overlays: Overlay<State>[]
): Overlay<State>[] => {
  const patchesField = getPatchesField(view);
  if (!patchesField) return overlays;

  const filtered = overlays.filter((o) => o.field === patchesField);
  return filtered.length > 0 ? filtered : overlays;
};

const adjustBox = (
  [w, h]: Dimensions,
  [obtlx, obtly, obw, obh]: BoundingBox
): {
  center: Coordinates;
  box: BoundingBox;
} => {
  const ar = obw / obh;
  let [btlx, btly, bw, bh] = [obtlx, obtly, obw, obh];

  if (bw * w < MIN_PIXELS) {
    bw = MIN_PIXELS / w;
    bh = bw / ar;
    btlx = obtlx + obw / 2 - bw / 2;
    btly = obtly + obh / 2 - bh / 2;
  }

  if (bh * h < MIN_PIXELS) {
    bh = MIN_PIXELS / h;
    bw = bh * ar;
    btlx = obtlx + obw / 2 - bw / 2;
    btly = obtly + obh / 2 - bh / 2;
  }

  return {
    center: [obtlx + obw / 2, obtly + obh / 2],
    box: [btlx, btly, bw, bh],
  };
};

export const zoomToContent = <
  State extends FrameState | ImageState | VideoState
>(
  state: Readonly<State>,
  overlays: Overlay<State>[]
): State => {
  const points = overlays.map((o) => o.getPoints(state)).flat();
  const [iw, ih] = state.dimensions;
  let [w, h] = [iw, ih];
  const iAR = w / h;
  const {
    center: [cw, ch],
    box: [_, __, bw, bh],
  } = adjustBox([w, h], getContainingBox(points));

  const [___, ____, ww, wh] = state.windowBBox;
  const wAR = ww / wh;

  let scale = 1;
  let pan: Coordinates = [0, 0];
  const squeeze = 1 - state.options.zoomPad;

  if (wAR < iAR) {
    scale = Math.max(1, 1 / bw);
    w = ww * scale;
    h = w / iAR;
    if (!state.config.thumbnail && bh * h > wh) {
      scale = Math.max(1, (wh * scale) / (bh * h));
      w = ww * scale;
      h = w / iAR;
    }
  } else {
    scale = Math.max(1, 1 / bh);
    h = wh * scale;
    w = h * iAR;
    if (!state.config.thumbnail && bw * w > ww) {
      scale = Math.max(1, (ww * scale) / (bw * w));
      h = wh * scale;
      w = h * iAR;
    }
  }

  const marginX = (scale * ww - w) / 2;
  const marginY = (scale * wh - h) / 2;
  pan = [-w * cw - marginX + ww / 2, -h * ch - marginY + wh / 2];

  // Scale down and reposition for a centered patch with padding
  if ((w * squeeze > ww && h * squeeze > wh) || !state.config.thumbnail) {
    scale = squeeze * scale;
    pan[0] = pan[0] * squeeze + (bw * w * (1 - squeeze)) / 2;
    pan[1] = pan[1] * squeeze + (bh * h * (1 - squeeze)) / 2;
  }

  pan = snapBox(scale, pan, [ww, wh], [iw, ih], !state.config.thumbnail);

  return mergeUpdates(state, { scale: scale, pan } as Partial<State>);
};

export const zoomAspectRatio = (
  sample: object,
  schema: Schema,
  mediaAspectRatio: number
): number => {
  let points = [];

  const recurse = (data: object, follow: boolean, prefix = "") => {
    Object.entries(data).forEach(([field, label]) => {
      const docType = getCls(prefix + field, schema);
      if (label && docType in POINTS_FROM_FO) {
        points = [...points, ...POINTS_FROM_FO[docType](label)];
        return;
      }

      if (
        follow &&
        getFieldInfo(prefix + field, schema)?.ftype ===
          DYNAMIC_EMBEDDED_DOCUMENT_FIELD
      ) {
        recurse(label, false, `${field}.`);
      }
    });
  };

  recurse(sample, true);

  let [_, __, width, height] = getContainingBox(points);

  if (width === 0 || height === 0) {
    if (width === height) {
      width = 1;
      height = 1;
    } else if (height === 0) {
      height = width;
    } else {
      width = height;
    }
  }
  return (width / height) * mediaAspectRatio;
};

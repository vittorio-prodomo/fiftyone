/**
 * Copyright 2017-2026, Voxel51, Inc.
 */

import DetectionOverlay from "./overlays/detection";
import HeatmapOverlay from "./overlays/heatmap";
import SegmentationOverlay from "./overlays/segmentation";
import { Overlay } from "./overlays/base";
import { BaseState } from "./state";
import { RENDER_STATUS_PAINTED } from "./worker/shared";

/**
 * Composites bitmap overlays (segmentation masks, heatmaps, detection masks)
 * into a single OffscreenCanvas so the main render loop only needs one
 * drawImage call instead of N.
 */
export class BitmapCompositor<State extends BaseState> {
  private canvas: OffscreenCanvas | null = null;
  private ctx: OffscreenCanvasRenderingContext2D | null = null;

  // Invalidation tracking
  private cachedOverlayIds: string | null = null;
  private cachedAlpha: number | null = null;
  private cachedDimensions: string | null = null;
  private cachedColoring: string | null = null;

  /**
   * Returns true if the given overlay is a bitmap overlay (has a mask/map
   * bitmap that we composite).
   */
  isBitmapOverlay(overlay: Overlay<State>, _state: Readonly<State>): boolean {
    if (overlay instanceof SegmentationOverlay) {
      return true;
    }
    if (overlay instanceof HeatmapOverlay) {
      return true;
    }
    if (overlay instanceof DetectionOverlay) {
      const label = (overlay as DetectionOverlay<State>).label;
      return Boolean(
        label.mask?.bitmap?.width &&
          label._renderStatus === RENDER_STATUS_PAINTED
      );
    }
    return false;
  }

  /**
   * Returns true if the current composite is still valid (no rebuild needed).
   */
  isValid(state: Readonly<State>, overlays: Overlay<State>[]): boolean {
    if (!this.canvas || !this.ctx) {
      return false;
    }

    const dims = state.dimensions
      ? `${state.dimensions[0]}x${state.dimensions[1]}`
      : "";

    if (dims !== this.cachedDimensions) {
      return false;
    }

    if (state.options.alpha !== this.cachedAlpha) {
      return false;
    }

    // Build a fingerprint of visible bitmap overlays
    const ids = this.getOverlayFingerprint(overlays, state);
    if (ids !== this.cachedOverlayIds) {
      return false;
    }

    const coloringKey = `${state.options.coloring.seed}-${state.options.coloring.by}`;
    if (coloringKey !== this.cachedColoring) {
      return false;
    }

    return true;
  }

  /**
   * Rebuilds the composite by drawing all bitmap overlays onto a single
   * OffscreenCanvas at image resolution.
   */
  rebuild(state: Readonly<State>, overlays: Overlay<State>[]): void {
    if (!state.dimensions) {
      return;
    }

    const [imgW, imgH] = state.dimensions;

    if (
      !this.canvas ||
      this.canvas.width !== imgW ||
      this.canvas.height !== imgH
    ) {
      this.canvas = new OffscreenCanvas(imgW, imgH);
      this.ctx = this.canvas.getContext("2d");
    }

    this.ctx.clearRect(0, 0, imgW, imgH);
    this.ctx.globalAlpha = state.options.alpha;
    this.ctx.imageSmoothingEnabled = false;

    // Draw in reverse order (back-to-front), matching the main render loop
    for (let index = overlays.length - 1; index >= 0; index--) {
      const overlay = overlays[index];
      if (!overlay.isShown(state)) {
        continue;
      }
      this.drawOverlayToComposite(overlay, state, imgW, imgH);
    }

    this.ctx.globalAlpha = 1;

    // Update cache keys
    this.cachedAlpha = state.options.alpha;
    this.cachedDimensions = `${imgW}x${imgH}`;
    this.cachedOverlayIds = this.getOverlayFingerprint(overlays, state);
    this.cachedColoring = `${state.options.coloring.seed}-${state.options.coloring.by}`;
  }

  /**
   * Draws the single composite bitmap onto the main canvas at the correct
   * canvasBBox position.
   */
  drawToCanvas(ctx: CanvasRenderingContext2D, state: Readonly<State>): void {
    if (!this.canvas || !state.canvasBBox) {
      return;
    }

    const [tlx, tly, w, h] = state.canvasBBox;
    ctx.drawImage(this.canvas, tlx, tly, w, h);
  }

  /**
   * Releases OffscreenCanvas resources.
   */
  dispose(): void {
    this.canvas = null;
    this.ctx = null;
    this.cachedOverlayIds = null;
    this.cachedAlpha = null;
    this.cachedDimensions = null;
    this.cachedColoring = null;
  }

  private drawOverlayToComposite(
    overlay: Overlay<State>,
    _state: Readonly<State>,
    imgW: number,
    imgH: number
  ): void {
    if (overlay instanceof SegmentationOverlay) {
      const label = (overlay as SegmentationOverlay<State>).label;
      if (label.mask?.bitmap?.width) {
        this.ctx.drawImage(label.mask.bitmap, 0, 0, imgW, imgH);
      }
    } else if (overlay instanceof HeatmapOverlay) {
      const label = (overlay as HeatmapOverlay<State>).label;
      if (label.map?.bitmap?.width) {
        this.ctx.drawImage(label.map.bitmap, 0, 0, imgW, imgH);
      }
    } else if (overlay instanceof DetectionOverlay) {
      const label = (overlay as DetectionOverlay<State>).label;
      if (
        label.mask?.bitmap?.width &&
        label._renderStatus === RENDER_STATUS_PAINTED &&
        label.bounding_box
      ) {
        const [bx, by, bw, bh] = label.bounding_box;
        this.ctx.drawImage(
          label.mask.bitmap,
          bx * imgW,
          by * imgH,
          bw * imgW,
          bh * imgH
        );
      }
    }
  }

  private getOverlayFingerprint(
    overlays: Overlay<State>[],
    state: Readonly<State>
  ): string {
    const parts: string[] = [];
    for (let i = 0; i < overlays.length; i++) {
      const overlay = overlays[i];
      if (!this.isBitmapOverlay(overlay, state) || !overlay.isShown(state)) {
        continue;
      }
      // Use label id + bitmap dimensions as fingerprint
      const id = overlay.label?.id ?? `idx-${i}`;
      if (overlay instanceof DetectionOverlay) {
        const bmp = (overlay as DetectionOverlay<State>).label.mask?.bitmap;
        parts.push(`d:${id}:${bmp?.width ?? 0}x${bmp?.height ?? 0}`);
      } else if (overlay instanceof SegmentationOverlay) {
        const bmp = (overlay as SegmentationOverlay<State>).label.mask?.bitmap;
        parts.push(`s:${id}:${bmp?.width ?? 0}x${bmp?.height ?? 0}`);
      } else if (overlay instanceof HeatmapOverlay) {
        const bmp = (overlay as HeatmapOverlay<State>).label.map?.bitmap;
        parts.push(`h:${id}:${bmp?.width ?? 0}x${bmp?.height ?? 0}`);
      }
    }
    return parts.join("|");
  }
}

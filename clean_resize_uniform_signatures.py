#!/usr/bin/env python3
# save as 2_clean_resize_uniform_signatures.py

from pathlib import Path
import cv2
import numpy as np

ROOT = Path("adaptive_threshold")
OUT  = Path("signature_clean")

# --- Cleanup tunables ---
MIN_AREA_RATIO = 0.003       # remove tiny components
MARGIN_BELOW_SIGNATURE = 4   # pixels kept below signature
MARGIN_ABOVE_SIGNATURE = 6   # pixels kept above signature

# Top border line filter
TOP_BAND = 0.12              # treat lines in top 12% as border lines
THIN_FRAC = 0.03             # "thin" height ≤ 3% of H
WIDE_FRAC = 0.60             # "long"  width ≥ 60% of W

# Left border line filter (new)
LEFT_BAND = 0.10             # treat the left 10% as border band
V_THIN_FRAC = 0.03           # "thin" width ≤ 3% of W
V_LONG_FRAC = 0.70           # "long" height ≥ 70% of H
LEFT_SAFETY_MARGIN = 6       # extra pixels to trim left of signature bbox
# -----------------------------------------------

# --- Uniformity / resizing tunables ---
CANVAS_H = 256               # final canvas height
CANVAS_W = 1024              # final canvas width
MARGIN_PX = 12               # margin around the cropped signature on the canvas
UPSCALE_INTERP = cv2.INTER_CUBIC
DOWNSCALE_INTERP = cv2.INTER_AREA
# -------------------------------------

OUT.mkdir(parents=True, exist_ok=True)
count = 0

for src_path in sorted(ROOT.rglob("*.png")):
    bin_bw = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)  # black on white (0/255)
    if bin_bw is None:
        print(f"skip (read error): {src_path}")
        continue

    H, W = bin_bw.shape

    # Foreground as white for CC analysis
    th = cv2.bitwise_not(bin_bw)

    # Light close to connect strokes
    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    min_area = int(MIN_AREA_RATIO * H * W)

    keep = np.zeros_like(th)
    boxes = []
    for i in range(1, n):  # 0 is background
        x, y, w, h, area = stats[i]

        # --- TOP-LINE FILTER: thin + long component near/top edge ---
        touches_top = (y <= 1)
        in_top_band = (y < int(TOP_BAND * H))
        is_thin = (h <= int(THIN_FRAC * H))
        is_long = (w >= int(WIDE_FRAC * W))
        if (touches_top or in_top_band) and is_thin and is_long:
            continue

        # --- LEFT-LINE FILTER: thin + tall component near/at left edge ---
        touches_left = (x <= 1)
        in_left_band = (x < int(LEFT_BAND * W))
        is_v_thin = (w <= int(V_THIN_FRAC * W))
        is_v_long = (h >= int(V_LONG_FRAC * H))
        if (touches_left or in_left_band) and is_v_thin and is_v_long:
            continue

        # Keep big-ish components (signature)
        if area >= min_area or h > 0.08 * H or w > 0.10 * W:
            keep[labels == i] = 255
            boxes.append((x, y, w, h))

    if not boxes:
        # Fallback: keep everything (rare)
        keep = th
        boxes = [(0, 0, W, H)]

    # Cut above & below signature
    y_top = max(0, min(y for (_, y, _, _) in boxes) - MARGIN_ABOVE_SIGNATURE)
    y_bottom = max(y + h for (_, y, _, h) in boxes)
    y_cut = min(H, y_bottom + MARGIN_BELOW_SIGNATURE)
    keep[:y_top, :] = 0
    keep[y_cut:, :] = 0

    # Extra safety: trim far-left area just left of the kept signature bbox
    x_left = max(0, min(x for (x, _, _, _) in boxes) - LEFT_SAFETY_MARGIN)
    keep[:, :x_left] = 0

    # Final binary mask of signature (white on black)
    mask = keep

    # --- Tight crop around the signature ---
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        # nothing left; write a white canvas and continue
        canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
        dest_path = OUT / src_path.relative_to(ROOT)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest_path), canvas)
        print(f"{dest_path} (empty after cleanup)")
        count += 1
        continue

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # Add a small margin before resizing
    y0 = max(0, y0 - MARGIN_PX)
    y1 = min(H - 1, y1 + MARGIN_PX)
    x0 = max(0, x0 - MARGIN_PX)
    x1 = min(W - 1, x1 + MARGIN_PX)

    cropped_mask = mask[y0:y1+1, x0:x1+1]
    cropped_img  = cv2.bitwise_not(cropped_mask)  # back to black-on-white

    h, w = cropped_img.shape

    # --- Resize to fit the target canvas while keeping aspect ratio ---
    max_h = CANVAS_H - 2 * MARGIN_PX
    max_w = CANVAS_W - 2 * MARGIN_PX
    scale = max(1e-6, min(max_h / h, max_w / w))

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = UPSCALE_INTERP if scale > 1.0 else DOWNSCALE_INTERP
    resized = cv2.resize(cropped_img, (new_w, new_h), interpolation=interp)

    # --- Place on a white canvas, centered ---
    canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    y_start = (CANVAS_H - new_h) // 2
    x_start = (CANVAS_W - new_w) // 2
    canvas[y_start:y_start+new_h, x_start:x_start+new_w] = resized

    dest_path = OUT / src_path.relative_to(ROOT)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest_path), canvas):
        print(f"failed to write: {dest_path}")
        continue

    print(dest_path)
    count += 1

print(f"Cleaned & standardized {count} PNGs into '{OUT.resolve()}'")

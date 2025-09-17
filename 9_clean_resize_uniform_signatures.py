from pathlib import Path
import cv2
import numpy as np

ROOT = Path("adaptive_threshold_stabilized")
OUT  = Path("signature_clean")

MIN_AREA_RATIO = 0.003
MARGIN_BELOW_SIGNATURE = 4
MARGIN_ABOVE_SIGNATURE = 6

TOP_BAND = 0.12
THIN_FRAC = 0.03
WIDE_FRAC = 0.60

LEFT_BAND = 0.10
V_THIN_FRAC = 0.03
V_LONG_FRAC = 0.70
LEFT_SAFETY_MARGIN = 6

CANVAS_H = 256
CANVAS_W = 1024
MARGIN_PX = 12
UPSCALE_INTERP = cv2.INTER_CUBIC
DOWNSCALE_INTERP = cv2.INTER_AREA

OUT.mkdir(parents=True, exist_ok=True)
count = 0

for src_path in sorted(ROOT.rglob("*.png")):
    bin_bw = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if bin_bw is None:
        print(f"skip (read error): {src_path}")
        continue

    H, W = bin_bw.shape

    th = cv2.bitwise_not(bin_bw)

    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    min_area = int(MIN_AREA_RATIO * H * W)

    keep = np.zeros_like(th)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]

        touches_top = (y <= 1)
        in_top_band = (y < int(TOP_BAND * H))
        is_thin = (h <= int(THIN_FRAC * H))
        is_long = (w >= int(WIDE_FRAC * W))
        if (touches_top or in_top_band) and is_thin and is_long:
            continue

        touches_left = (x <= 1)
        in_left_band = (x < int(LEFT_BAND * W))
        is_v_thin = (w <= int(V_THIN_FRAC * W))
        is_v_long = (h >= int(V_LONG_FRAC * H))
        if (touches_left or in_left_band) and is_v_thin and is_v_long:
            continue

        if area >= min_area or h > 0.08 * H or w > 0.10 * W:
            keep[labels == i] = 255
            boxes.append((x, y, w, h))

    if not boxes:
        keep = th
        boxes = [(0, 0, W, H)]

    y_top = max(0, min(y for (_, y, _, _) in boxes) - MARGIN_ABOVE_SIGNATURE)
    y_bottom = max(y + h for (_, y, _, h) in boxes)
    y_cut = min(H, y_bottom + MARGIN_BELOW_SIGNATURE)
    keep[:y_top, :] = 0
    keep[y_cut:, :] = 0

    x_left = max(0, min(x for (x, _, _, _) in boxes) - LEFT_SAFETY_MARGIN)
    keep[:, :x_left] = 0

    mask = keep

    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        canvas = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
        dest_path = OUT / src_path.relative_to(ROOT)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest_path), canvas)
        print(f"{dest_path} (empty after cleanup)")
        count += 1
        continue

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(0, y0 - MARGIN_PX)
    y1 = min(H - 1, y1 + MARGIN_PX)
    x0 = max(0, x0 - MARGIN_PX)
    x1 = min(W - 1, x1 + MARGIN_PX)

    cropped_mask = mask[y0:y1+1, x0:x1+1]
    cropped_img  = cv2.bitwise_not(cropped_mask)

    h, w = cropped_img.shape

    max_h = CANVAS_H - 2 * MARGIN_PX
    max_w = CANVAS_W - 2 * MARGIN_PX
    scale = max(1e-6, min(max_h / h, max_w / w))

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = UPSCALE_INTERP if scale > 1.0 else DOWNSCALE_INTERP
    resized = cv2.resize(cropped_img, (new_w, new_h), interpolation=interp)

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
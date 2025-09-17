from pathlib import Path
import cv2
import numpy as np

ROOT = Path("signature_clean_horizontal")
OUT = Path("signature_clean_horizontal_smoothed")

UPSCALE = 3
GAP_CLOSE = 3
LINE_BRIDGE = 5
SMOOTH_SIGMA = 1.2
THIN_PIXELS = 0.6

OUT.mkdir(parents=True, exist_ok=True)


def ensure_binary_white_ink(img):
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = (img[..., 3].astype(np.float32) / 255.0)[..., None]
        rgb = img[..., :3].astype(np.float32)
        white = np.full_like(rgb, 255.0, dtype=np.float32)
        img_rgb = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(th == 255) > th.size // 2:
        th = 255 - th
    return th


def _odd_ksize_from_sigma(sigma: float) -> int:
    k = max(3, int(round(sigma * 6)))
    return k if (k % 2) == 1 else k + 1


def _signed_distance(mask255: np.ndarray) -> np.ndarray:
    m = (mask255 > 0).astype(np.uint8)
    if m.max() == 0:
        return -cv2.distanceTransform(1 - m, cv2.DIST_L2, 3).astype(np.float32)
    if m.min() == 1:
        return cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3).astype(np.float32)
    return din - dout


def process(mask_white_on_black: np.ndarray) -> np.ndarray:
    h, w = mask_white_on_black.shape[:2]
    if UPSCALE > 1:
        hi = cv2.resize(mask_white_on_black, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_NEAREST)
    else:
        hi = mask_white_on_black
    if GAP_CLOSE >= 2:
        hi = cv2.morphologyEx(hi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GAP_CLOSE, GAP_CLOSE)), iterations=1)
    if LINE_BRIDGE >= 3:
        hi = cv2.morphologyEx(hi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_BRIDGE, 1)), iterations=1)
        hi = cv2.morphologyEx(hi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_BRIDGE)), iterations=1)
    k = _odd_ksize_from_sigma(SMOOTH_SIGMA * max(1, UPSCALE))
    hi = cv2.GaussianBlur(hi, (k, k), SMOOTH_SIGMA * max(1, UPSCALE))
    _, hi = cv2.threshold(hi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if THIN_PIXELS > 0:
        sdf = _signed_distance(hi)
        offset = THIN_PIXELS * max(1, UPSCALE)
        hi = (sdf >= offset).astype(np.uint8) * 255
    lo = cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA) if UPSCALE > 1 else hi
    return np.where(lo >= 128, 255, 0).astype(np.uint8)


count = 0
for src_path in sorted(ROOT.rglob("*.png")):
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"skip (read error): {src_path}")
        continue
    bin_white = ensure_binary_white_ink(img)
    out = process(bin_white)
    dest_path = OUT / src_path.relative_to(ROOT)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest_path), out):
        print(f"failed to write: {dest_path}")
        continue
    print(dest_path)
    count += 1

print(f"Smoothed {count} PNGs into '{OUT.resolve()}'")
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("signature_clean_horizontal")
OUT  = Path("signature_clean_horizontal_smoothed")

EPSILON_RATIO   = 0.008  
MIN_AREA_RATIO  = 0.0005 
POST_MEDIAN_K   = 0 

OUT.mkdir(parents=True, exist_ok=True)

def ensure_binary_ink_black(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(th) > 127:
        th = 255 - th

    return th

def smooth_contours(binary_black_ink: np.ndarray) -> np.ndarray:
    h, w = binary_black_ink.shape[:2]
    min_area = max(1.0, MIN_AREA_RATIO * (h * w))

    work = 255 - binary_black_ink

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(work)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perim = cv2.arcLength(cnt, True)
        eps = max(1.0, EPSILON_RATIO * perim)
        approx = cv2.approxPolyDP(cnt, eps, True)
        cv2.fillPoly(filled, [approx], 255)

    out = 255 - filled

    if POST_MEDIAN_K in (3, 5):
        out = cv2.medianBlur(out, POST_MEDIAN_K)
        _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)

    return out

count = 0
for src_path in sorted(ROOT.rglob("*.png")):
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"skip (read error): {src_path}")
        continue

    bin_img = ensure_binary_ink_black(img)
    smoothed = smooth_contours(bin_img)

    dest_path = OUT / src_path.relative_to(ROOT)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest_path), smoothed):
        print(f"failed to write: {dest_path}")
        continue

    print(dest_path)
    count += 1

print(f"Contour-smoothed {count} PNGs into '{OUT.resolve()}'")

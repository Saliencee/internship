from pathlib import Path
import cv2
import numpy as np

ROOT = Path("adaptive_threshold")
OUT  = Path("adaptive_threshold_stabilized")

CLOSE_K = 3
OPEN_K  = 3   
CLOSE_ITER = 1

OUT.mkdir(parents=True, exist_ok=True)
count = 0

for src_path in sorted(ROOT.rglob("*.png")):
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"skip (read error): {src_path}")
        continue
    if np.unique(img).size > 2:
        _, bin_bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        bin_bw = img
    inv = cv2.bitwise_not(bin_bw)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    tmp = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, k_close, iterations=CLOSE_ITER)
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN,  k_open,  iterations=1)

    tmp = cv2.medianBlur(tmp, 3)
    out = cv2.bitwise_not(tmp)

    dest_path = OUT / src_path.relative_to(ROOT)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest_path), out):
        print(f"failed to write: {dest_path}")
        continue

    print(dest_path)
    count += 1

print(f"Stabilized {count} PNGs into '{OUT.resolve()}'")

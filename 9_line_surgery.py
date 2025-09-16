from pathlib import Path
import cv2
import numpy as np

ROOT = Path("signature_clean")
OUT  = Path("signature_clean_horizontal")

MIN_REL_LEN = 0.35
THICKNESS   = 3 

OUT.mkdir(parents=True, exist_ok=True)

def binarize(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = 255 - th
    return th

def remove_horizontal_rules(bin_img: np.ndarray,
                            min_rel_len: float = MIN_REL_LEN,
                            thickness: int = THICKNESS) -> np.ndarray:
    h, w = bin_img.shape[:2]
    klen = max(int(w * float(min_rel_len)), 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, int(thickness)))
    lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.subtract(bin_img, lines)
    return cleaned

count = 0
for src_path in sorted(ROOT.rglob("*.png")):
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"skip (read error): {src_path}")
        continue

    bin_img = binarize(img)
    out_img = remove_horizontal_rules(bin_img)

    dest_path = OUT / src_path.relative_to(ROOT)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest_path), out_img):
        print(f"failed to write: {dest_path}")
        continue

    print(dest_path)
    count += 1

print(f"Processed {count} PNGs into '{OUT.resolve()}'")

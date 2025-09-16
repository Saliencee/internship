from pathlib import Path
import cv2
import numpy as np

ROOT = Path("signature_clean_horizontal")
OUT  = Path("skeletons_clean")

OUT.mkdir(parents=True, exist_ok=True)

def binarize(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = 255 - th
    return th

def morphological_skeleton(binary_img: np.ndarray) -> np.ndarray:
    img = binary_img.copy()
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel

count = 0
for src_path in sorted(ROOT.rglob("*.png")):
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"skip (read error): {src_path}")
        continue

    bin_img = binarize(img)
    skel = morphological_skeleton(bin_img)

    dest_path = OUT / src_path.relative_to(ROOT)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest_path), skel):
        print(f"failed to write: {dest_path}")
        continue

    print(dest_path)
    count += 1

print(f"Skeletonized {count} PNGs into '{OUT.resolve()}'")

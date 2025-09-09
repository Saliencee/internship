from pathlib import Path
import cv2

ROOT = Path("gaussian_blur")
OUT  = Path("adaptive_threshold")

BLOCK_SIZE = 15
C = 3
THRESH_TYPE = cv2.THRESH_BINARY
ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

b = BLOCK_SIZE if BLOCK_SIZE % 2 else BLOCK_SIZE + 1
OUT.mkdir(parents=True, exist_ok=True)
count = 0

for src_path in sorted(ROOT.rglob("*.png")):
    gray = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"skip (read error): {src_path}")
        continue

    bin_img = cv2.adaptiveThreshold(
        gray, 255, ADAPTIVE_METHOD, THRESH_TYPE, b, C
    )

    dest_path = OUT / src_path.relative_to(ROOT)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest_path), bin_img):
        print(f"failed to write: {dest_path}")
        continue

    print(dest_path)
    count += 1

print(f"Adaptive-thresholded {count} PNGs into '{OUT.resolve()}'")
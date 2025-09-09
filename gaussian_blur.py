from pathlib import Path
import cv2

ROOT = Path("greyscale") 
OUT  = Path("gaussian_blur")  
K = 5                       

k = K if K % 2 else K + 1
OUT.mkdir(parents=True, exist_ok=True)

count = 0
for src_path in sorted(ROOT.rglob("*.png")):
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"skip (read error): {src_path}")
        continue
    blurred = cv2.GaussianBlur(img, (k, k), 0)

    dest_path = OUT / src_path.relative_to(ROOT)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest_path), blurred):
        print(f"failed to write: {dest_path}")
        continue

    print(dest_path)
    count += 1

print(f"Blurred {count} PNGs into '{OUT.resolve()}'")
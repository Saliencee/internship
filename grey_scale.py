from pathlib import Path
import cv2

cropped_dir = Path("sample_dataset_portrait_cropped")
org_root = Path("sample_dataset_portrait_cropped_org")

out_root = Path("greyscale")
out_root.mkdir(exist_ok=True)

def save_gray(img_path: Path, root: Path) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Skipped (read error): {img_path}")
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rel = img_path.relative_to(root)
    dest = out_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), gray)
    print(dest)
    return 1

count = 0
for p in sorted(cropped_dir.glob("*.png")):
    count += save_gray(p, cropped_dir)

for p in sorted(org_root.rglob("*.png")):
    count += save_gray(p, org_root)

print(f"Saved {count} images into '{out_root}/'")
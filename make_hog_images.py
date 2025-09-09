#!/usr/bin/env python3
# make_hog_only.py
# Simple: produce HOG-only images for your dataset into one folder inside "Internship".

from pathlib import Path
import numpy as np
from skimage import io, color, exposure
from skimage.feature import hog

# ---- Paths (edit if needed) ----
BASE_SIGNATURES = Path("/Users/matthieutohme/Desktop/internship/signature_clean")
OUT_ROOT        = Path("/Users/matthieutohme/Desktop/Internship/HOG_only")
# --------------------------------

# HOG parameters (good defaults for signatures)
ORIENTATIONS     = 9
PIXELS_PER_CELL  = (8, 8)
CELLS_PER_BLOCK  = (2, 2)
BLOCK_NORM       = "L2-Hys"
TRANSFORM_SQRT   = True

def to_gray_unit(img):
    """Return grayscale float image in [0,1]."""
    if img.ndim == 3:
        img = color.rgb2gray(img)
    elif img.ndim == 2:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return img

def save_hog_only(image_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = io.imread(str(image_path))
    gray = to_gray_unit(img)

    # Compute HOG with visualization
    _, hog_img = hog(
        gray,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm=BLOCK_NORM,
        transform_sqrt=TRANSFORM_SQRT,
        visualize=True,
        feature_vector=True,
    )

    # Rescale for visibility and save as uint8 PNG
    hog_vis = exposure.rescale_intensity(hog_img, in_range=(0, hog_img.max() or 1.0), out_range=(0, 255))
    hog_vis = hog_vis.astype(np.uint8)

    out_path = out_dir / f"{image_path.stem}_hog.png"
    io.imsave(str(out_path), hog_vis, check_contrast=False)

def main():
    if not BASE_SIGNATURES.exists():
        raise SystemExit(f"[ERROR] Base folder not found: {BASE_SIGNATURES}")

    groups = {
        "fraud":  sorted(BASE_SIGNATURES.glob("fraud_image_*.png")),
        "valid":  sorted(BASE_SIGNATURES.glob("valid_image_*.png")),
        "crop_1": sorted((BASE_SIGNATURES / "crop_1").glob("org_image_*.png")),
        "crop_2": sorted((BASE_SIGNATURES / "crop_2").glob("org_image_*.png")),
    }

    total = 0
    for name, paths in groups.items():
        print(f"Processing {name}: {len(paths)} images")
        out_dir = OUT_ROOT / name
        for p in paths:
            try:
                save_hog_only(p, out_dir)
                total += 1
            except Exception as e:
                print(f"[WARN] Skipped {p.name}: {e}")

    if total == 0:
        print("[WARN] No images found matching expected patterns.")
    else:
        print(f"Done. Saved {total} HOG images to: {OUT_ROOT}")

if __name__ == "__main__":
    main()
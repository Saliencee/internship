#!/usr/bin/env python3
"""
skeletonize.py
Skeletonize a cleaned/binary signature image (strokes should be white on black).

Usage:
  python skeletonize.py --input /path/to/image_or_dir --out /path/to/outdir

If the input is not yet binary or strokes are black-on-white, this script will
binarize and invert automatically.
"""
import argparse, cv2, numpy as np
from pathlib import Path

def binarize(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure strokes are white on black
    if np.mean(th) > 127:
        th = 255 - th
    return th

def morphological_skeleton(binary_img):
    """Iterative morphological skeletonization. Input must be 0/255 with strokes=255."""
    img = binary_img.copy()
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)  # open(eroded)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel

def process_image_path(in_file: Path, out_dir: Path):
    img = cv2.imread(str(in_file))
    if img is None:
        return
    bin_img = binarize(img)
    skel = morphological_skeleton(bin_img)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "01_binary_for_skeleton.png"), bin_img)
    cv2.imwrite(str(out_dir / "02_skeleton.png"), skel)

def process_directory(input_dir: Path, out_root: Path):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for p in sorted(input_dir.rglob("*")):
        if p.suffix.lower() in exts:
            rel = p.relative_to(input_dir)
            out_dir = out_root / rel.parent / rel.stem
            process_image_path(p, out_dir)

def main():
    ap = argparse.ArgumentParser(description="Signature skeletonization.")
    ap.add_argument("--input", required=True, help="Image file or directory")
    ap.add_argument("--out",   required=True, help="Output directory")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    if inp.is_dir():
        process_directory(inp, out)
    else:
        out_dir = out / inp.stem
        process_image_path(inp, out_dir)

if __name__ == "__main__":
    main()

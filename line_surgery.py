#!/usr/bin/env python3
"""
line_surgery.py
Remove long underlines/rules and other long straight segments before thinning.
Also performs a *gentle* stitching (directional closing) to reconnect tiny breaks.

Usage:
  python line_surgery.py --input /path/to/image_or_dir --out /path/to/outdir
  # optional tuning:
  python line_surgery.py --input ... --out ... --min_rel_len 0.35 --length_rel 0.35 --thickness 3

Input can be a single image file or a directory (processed recursively).
Outputs per-image (for auditing):
  01_binary.png
  02_removed_horizontal.png
  03_after_horizontal.png
  04_removed_hough.png
  05_after_hough.png
  06_after_gap_closing.png     <-- final output of this step (recommended for next stage)
"""
import os, math, argparse, cv2, numpy as np
from pathlib import Path

def binarize(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu threshold to binary
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure strokes are white on black
    if np.mean(th) > 127:
        th = 255 - th
    return th

def remove_horizontal_rules(bin_img, min_rel_len=0.40, thickness=3):
    """Strip very long horizontal lines (underlines/page rules) via morphological opening."""
    h, w = bin_img.shape[:2]
    klen = max(int(w * float(min_rel_len)), 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, int(thickness)))
    lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.subtract(bin_img, lines)
    return cleaned, lines

def remove_long_straights_via_hough(bin_img, length_rel=0.30, canny1=50, canny2=150):
    """Remove remaining long straight segments at any angle using HoughLinesP."""
    h, w = bin_img.shape[:2]
    edges = cv2.Canny(bin_img, canny1, canny2)
    min_len = int(max(h, w) * float(length_rel))
    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                             minLineLength=min_len, maxLineGap=10)
    mask = np.zeros_like(bin_img)
    if linesP is not None:
        for x1, y1, x2, y2 in linesP[:,0,:]:
            length = math.hypot(x2-x1, y2-y1)
            if length < min_len:
                continue
            cv2.line(mask, (x1,y1), (x2,y2), 255, 2, cv2.LINE_AA)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)
        removed = cv2.bitwise_and(bin_img, mask)
        cleaned = cv2.subtract(bin_img, removed)
        return cleaned, removed
    return bin_img.copy(), np.zeros_like(bin_img)

def small_gap_closing_directional(bin_img, L=7, angles=(0, 15, -15, 30, -30)):
    """Reconnect tiny breaks along near-straight directions using short, linear kernels."""
    work = bin_img.copy()
    for ang in angles:
        k = np.zeros((L, L), np.uint8)
        cx = cy = L//2
        rad = math.radians(ang)
        dx = int((L//2) * math.cos(rad))
        dy = int((L//2) * math.sin(rad))
        cv2.line(k, (cx-dx, cy-dy), (cx+dx, cy+dy), 1, 1)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k, iterations=1)
    return work

def process_image_path(in_file: Path, out_dir: Path, min_rel_len=0.35, thickness=3, length_rel=0.35):
    img = cv2.imread(str(in_file))
    if img is None:
        return
    bin_img = binarize(img)
    step1, lines_h = remove_horizontal_rules(bin_img, min_rel_len=min_rel_len, thickness=thickness)
    step2, lines_hough = remove_long_straights_via_hough(step1, length_rel=length_rel)
    step2 = cv2.medianBlur(step2, 3)
    step3 = small_gap_closing_directional(step2, L=7, angles=(0, 15, -15, 30, -30))
    # tiny global closing to heal 1‑px notches (light)
    step3b = cv2.morphologyEx(step3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "01_binary.png"), bin_img)
    cv2.imwrite(str(out_dir / "02_removed_horizontal.png"), lines_h)
    cv2.imwrite(str(out_dir / "03_after_horizontal.png"), step1)
    cv2.imwrite(str(out_dir / "04_removed_hough.png"), lines_hough)
    cv2.imwrite(str(out_dir / "05_after_hough.png"), step2)
    cv2.imwrite(str(out_dir / "06_after_gap_closing.png"), step3b)

def process_directory(input_dir: Path, out_root: Path, **kwargs):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for p in sorted(input_dir.rglob("*")):
        if p.suffix.lower() in exts:
            rel = p.relative_to(input_dir)
            out_dir = out_root / rel.parent / rel.stem
            process_image_path(p, out_dir, **kwargs)

def main():
    ap = argparse.ArgumentParser(description="Line-surgery (remove long straight lines) and gentle stitching.")
    ap.add_argument("--input", required=True, help="Image file or directory")
    ap.add_argument("--out",   required=True, help="Output directory")
    ap.add_argument("--min_rel_len", type=float, default=0.35, help="Min relative width for horizontal-rule removal (0..1)")
    ap.add_argument("--thickness", type=int, default=3, help="Kernel thickness for horizontal-rule removal")
    ap.add_argument("--length_rel", type=float, default=0.35, help="Min relative length for Hough line removal (0..1)")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    if inp.is_dir():
        process_directory(inp, out, min_rel_len=args.min_rel_len, thickness=args.thickness, length_rel=args.length_rel)
    else:
        out_dir = out / inp.stem
        process_image_path(inp, out_dir, min_rel_len=args.min_rel_len, thickness=args.thickness, length_rel=args.length_rel)

if __name__ == "__main__":
    main()

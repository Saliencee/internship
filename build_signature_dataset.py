#!/usr/bin/env python3
"""
Builds the signature sample dataset from 30 PDFs:
- fraud image NN.pdf          (NN = 01..10) -> fraud_image_001..010.png
- valid image NN.pdf          (NN = 01..10) -> valid_image_011..020.png
- specimen NN.pdf (first page)-> org_image_001..010.png and org_image_011..020.png

Outputs:
- sample_dataset/ (40 PNGs)
- sample_dataset.zip

Usage (run from the folder that contains the PDFs):
    python3 build_signature_dataset.py
Options:
    --src <path>         Source folder (default: .)
    --out <path>         Output folder (default: sample_dataset)
    --dpi <int>          Render DPI for pdf2image (default: 300)
    --force              Overwrite existing PNGs/ZIP
    --no-zip             Skip zipping
"""
from pathlib import Path
import argparse
import sys
import shutil

def main():
    parser = argparse.ArgumentParser(description="Build automated signature-verification sample dataset.")
    parser.add_argument("--src", default=".", type=str, help="Folder with the 30 PDFs")
    parser.add_argument("--out", default="sample_dataset", type=str, help="Output folder for PNGs")
    parser.add_argument("--dpi", default=300, type=int, help="DPI for PDF rendering")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--no-zip", action="store_true", help="Do not create sample_dataset.zip")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out_dir = (Path.cwd() / args.out).resolve()

    # Lazy-import so we can print a clearer error if missing
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError
    except Exception as e:
        print("Error: pdf2image not installed. Install with:\n"
              "  python3 -m pip install pdf2image pillow\n"
              "On macOS, you may also need Poppler:\n"
              "  brew install poppler", file=sys.stderr)
        sys.exit(1)

    # Validate inputs exist
    missing = []
    for n in range(1, 11):
        n2 = f"{n:02d}"
        for kind in ("fraud image", "valid image", "specimen"):
            p = src / f"{kind} {n2}.pdf"
            if not p.exists():
                missing.append(str(p))
    if missing:
        print("ERROR: The following input PDFs were not found:", file=sys.stderr)
        for m in missing:
            print(" -", m, file=sys.stderr)
        sys.exit(2)

    # Prepare output directory
    if out_dir.exists():
        if args.force:
            shutil.rmtree(out_dir)
        else:
            # Keep existing directory but don't overwrite files unless --force for each save
            pass
    out_dir.mkdir(parents=True, exist_ok=True)

    # Conversion helpers
    def convert_first_page(pdf_path: Path):
        """Return the first page as a PIL Image at the desired DPI."""
        try:
            imgs = convert_from_path(str(pdf_path), dpi=args.dpi, first_page=1, last_page=1)
        except PDFInfoNotInstalledError:
            print("Poppler not found. On macOS, install with: brew install poppler", file=sys.stderr)
            sys.exit(3)
        if not imgs:
            raise RuntimeError(f"No pages found in {pdf_path}")
        return imgs[0]

    def save_png(image, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.force:
            # Skip if already exists unless --force
            return
        # PNG can store DPI in metadata
        image.save(str(path), format="PNG", dpi=(args.dpi, args.dpi))

    # Process all 10 indices
    print(f"Converting PDFs from: {src}")
    print(f"Writing PNGs to:     {out_dir}")
    print(f"DPI:                 {args.dpi}")

    for n in range(1, 11):
        n2 = f"{n:02d}"        # matches input file names
        idx_00 = f"{n:03d}"    # 001..010 (forged set)
        idx_01 = f"{n+10:03d}" # 011..020 (valid set)

        fraud_pdf = src / f"fraud image {n2}.pdf"
        valid_pdf = src / f"valid image {n2}.pdf"
        spec_pdf  = src / f"specimen {n2}.pdf"

        # Convert and save fraud sample
        fraud_png = out_dir / f"fraud_image_{idx_00}.png"
        fraud_img = convert_first_page(fraud_pdf)
        save_png(fraud_img, fraud_png)

        # Convert and save valid sample
        valid_png = out_dir / f"valid_image_{idx_01}.png"
        valid_img = convert_first_page(valid_pdf)
        save_png(valid_img, valid_png)

        # Convert specimen once, save two copies (org for forged + org for valid)
        spec_img = convert_first_page(spec_pdf)
        org_png_00 = out_dir / f"org_image_{idx_00}.png"
        org_png_01 = out_dir / f"org_image_{idx_01}.png"
        save_png(spec_img, org_png_00)
        save_png(spec_img, org_png_01)

    # Verify pairing
    print("\nVerifying pairings...")
    problems = []
    for n in range(1, 11):
        idx_00 = f"{n:03d}"
        idx_01 = f"{n+10:03d}"
        pair_00 = (out_dir / f"fraud_image_{idx_00}.png", out_dir / f"org_image_{idx_00}.png")
        pair_01 = (out_dir / f"valid_image_{idx_01}.png", out_dir / f"org_image_{idx_01}.png")

        for a, b in (pair_00, pair_01):
            if not a.exists() or not b.exists():
                problems.append((a.name, b.name))

    if problems:
        print("ERROR: Missing pair members:", file=sys.stderr)
        for a, b in problems:
            print(f" - {a} / {b}", file=sys.stderr)
        sys.exit(4)

    # Count files
    pngs = sorted([p for p in out_dir.glob("*.png")])
    count = len(pngs)
    print(f"OK: {count} PNGs created in {out_dir}")
    if count != 40:
        print("WARNING: Expected 40 PNGs.", file=sys.stderr)

    # Zip (unless skipped)
    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists() and args.force:
        zip_path.unlink(missing_ok=True)

    if not args.no_zip:
        print(f"Creating ZIP: {zip_path}")
        # Create ZIP that contains only the 40 PNGs at the root of the archive
        # by zipping the contents of out_dir
        shutil.make_archive(base_name=str(out_dir), format="zip", root_dir=str(out_dir), base_dir=".")
        print("Done.")

if __name__ == "__main__":
    main()
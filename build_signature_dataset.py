from pathlib import Path
import argparse
import sys
import shutil
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError

K_RANGE = range(1, 11)
KINDS = ("fraud image", "valid image", "specimen")

def convert_first_page(pdf_path: Path, dpi: int):
    try:
        imgs = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)
    except PDFInfoNotInstalledError:
        print("Poppler not found. On macOS, install with: brew install poppler", file=sys.stderr)
        sys.exit(3)
    if not imgs:
        raise RuntimeError(f"No pages found in {pdf_path}")
    return imgs[0]

def save_png(image, path: Path, dpi: int, force: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    image.save(str(path), format="PNG", dpi=(dpi, dpi))

def main():
    parser = argparse.ArgumentParser(description="Convert sample PDFs to PNGs (first page only).")
    parser.add_argument("--src", default=".", type=str, help="Folder with the 30 PDFs")
    parser.add_argument("--out", default="sample_dataset", type=str, help="Output folder for PNGs")
    parser.add_argument("--dpi", default=300, type=int, help="DPI for PDF rendering")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out_dir = (Path.cwd() / args.out).resolve()

    missing = []
    for n in K_RANGE:
        n2 = f"{n:02d}"
        for kind in KINDS:
            p = src / f"{kind} {n2}.pdf"
            if not p.exists():
                missing.append(str(p))
    if missing:
        print("ERROR: Missing input PDFs:", file=sys.stderr)
        for m in missing:
            print(f" - {m}", file=sys.stderr)
        sys.exit(2)

    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting from: {src}")
    print(f"Writing to:     {out_dir}")
    print(f"DPI:            {args.dpi}")

    for n in K_RANGE:
        n2 = f"{n:02d}"
        idx_00 = f"{n:03d}"
        idx_01 = f"{n+10:03d}"

        fraud_pdf = src / f"fraud image {n2}.pdf"
        valid_pdf = src / f"valid image {n2}.pdf"
        spec_pdf  = src / f"specimen {n2}.pdf"

        fraud_img = convert_first_page(fraud_pdf, args.dpi)
        save_png(fraud_img, out_dir / f"fraud_image_{idx_00}.png", args.dpi, args.force)

        valid_img = convert_first_page(valid_pdf, args.dpi)
        save_png(valid_img, out_dir / f"valid_image_{idx_01}.png", args.dpi, args.force)

        spec_img = convert_first_page(spec_pdf, args.dpi)
        save_png(spec_img, out_dir / f"org_image_{idx_00}.png", args.dpi, args.force)
        save_png(spec_img, out_dir / f"org_image_{idx_01}.png", args.dpi, args.force)

    problems = []
    for n in K_RANGE:
        idx_00 = f"{n:03d}"
        idx_01 = f"{n+10:03d}"
        pairs = [
            (out_dir / f"fraud_image_{idx_00}.png", out_dir / f"org_image_{idx_00}.png"),
            (out_dir / f"valid_image_{idx_01}.png", out_dir / f"org_image_{idx_01}.png"),
        ]
        for a, b in pairs:
            if not a.exists() or not b.exists():
                problems.append((a.name, b.name))

    if problems:
        print("ERROR: Missing pair members:", file=sys.stderr)
        for a, b in problems:
            print(f" - {a} / {b}", file=sys.stderr)
        sys.exit(4)

    pngs = sorted(out_dir.glob("*.png"))
    count = len(pngs)
    print(f"OK: {count} PNGs created in {out_dir}")
    if count != 40:
        print("WARNING: Expected 40 PNGs.", file=sys.stderr)

if __name__ == "__main__":
    main()
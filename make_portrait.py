from pathlib import Path
from PIL import Image
import argparse

def main():
    ap = argparse.ArgumentParser(description="Rotate PNGs to portrait orientation.")
    ap.add_argument("--src", default="sample_dataset", help="Folder with PNGs")
    ap.add_argument("--out", default="sample_dataset_portrait", help="Output folder")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for p in sorted(src.glob("*.png")):
        with Image.open(p) as im:
            w, h = im.size
            out_im = im if h >= w else im.transpose(Image.Transpose.ROTATE_90)

            save_kwargs = {}
            if "dpi" in im.info:
                save_kwargs["dpi"] = im.info["dpi"]

            out_im.save(out / p.name, **save_kwargs)
            count += 1

    print(f"Done. Wrote {count} images to {out.resolve()}")

if __name__ == "__main__":
    main()
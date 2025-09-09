from pathlib import Path
from PIL import Image

SRC = Path("sample_dataset_portrait")
DST = SRC.parent / "sample_dataset_portrait_cropped_org"

CROP_1 = (47, 830, 1121, 1260)
CROP_2 = (1209, 830, 2318, 1260)

OUT_1 = DST / "crop_1"
OUT_2 = DST / "crop_2"
OUT_1.mkdir(parents=True, exist_ok=True)
OUT_2.mkdir(parents=True, exist_ok=True)

count = 0
for img_path in sorted(SRC.glob("org_image_*.png")):
    name = img_path.name
    with Image.open(img_path) as im:
        w, h = im.size
        for box in (CROP_1, CROP_2):
            l, u, r, b = box
            if not (0 <= l < r <= w and 0 <= u < b <= h):
                raise ValueError(
                    f"Crop box {box} exceeds image bounds {w}x{h} for {name}"
                )

        im.crop(CROP_1).save(OUT_1 / name)
        im.crop(CROP_2).save(OUT_2 / name)
        count += 1

print(f"Cropped two regions for {count} org images.")
print(f"First region saved to:  {OUT_1.resolve()}")
print(f"Second region saved to: {OUT_2.resolve()}")
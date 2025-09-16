from pathlib import Path
from PIL import Image

SRC = Path("sample_dataset_portrait")
DST = SRC.parent / "sample_dataset_portrait_cropped"
DST.mkdir(exist_ok=True)

CROP_BOX = (516, 1200, 2000, 1800)

count = 0
for img_path in sorted(SRC.glob("*.png")):
    name = img_path.name
    if name.startswith(("valid_image_", "fraud_image_")):
        with Image.open(img_path) as im:
            cropped = im.crop(CROP_BOX)
            cropped.save(DST / name)
            count += 1

print(f"Cropped {count} images into: {DST.resolve()}")
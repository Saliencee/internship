# save as crop_valid_and_fraud.py
from pathlib import Path
from PIL import Image

SRC = Path("sample_dataset_portrait")
DST = SRC.parent / "sample_dataset_portrait_cropped"
DST.mkdir(exist_ok=True)

# Pillow expects (left, upper, right, lower)
CROP_BOX = (516, 1200, 2000, 1800)

count = 0
for img_path in sorted(SRC.glob("*.png")):
    name = img_path.name
    if name.startswith(("valid_image_", "fraud_image_")):
        with Image.open(img_path) as im:
            # Crop and save as PNG (lossless), same filename
            cropped = im.crop(CROP_BOX)
            cropped.save(DST / name)  # extension dictates format; PNG stays PNG
            count += 1

print(f"Cropped {count} images into: {DST.resolve()}")
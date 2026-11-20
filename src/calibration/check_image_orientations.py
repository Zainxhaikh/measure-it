
from pathlib import Path
from collections import Counter

from PIL import Image, ExifTags

IMAGE_DIR = Path("calibration/images")

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

dimensions = Counter()

print("CALIBRATION IMAGE ORIENTATION REPORT")
print("=" * 60)

for path in image_paths:
    try:
        with Image.open(path) as image:
            width, height = image.size

            exif = image.getexif()
            orientation = exif.get(274, 1)

            dimensions[(width, height)] += 1

            direction = (
                "PORTRAIT" if height > width
                else "LANDSCAPE" if width > height
                else "SQUARE"
            )

            print(
                f"{path.name} | "
                f"{width} x {height} | "
                f"{direction} | "
                f"EXIF orientation: {orientation}"
            )

    except Exception as error:
        print(f"ERROR: {path.name} | {error}")

print("\n" + "=" * 60)
print("IMAGE DIMENSION SUMMARY")

for size, count in dimensions.items():
    print(f"{size[0]} x {size[1]}: {count} images")

print("\nTotal images:", len(image_paths))

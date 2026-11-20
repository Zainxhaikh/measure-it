
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "dataset" / "images"
MASK_DIR = PROJECT_ROOT / "dataset" / "masks"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in image_extensions
)

passed = 0
failed = 0

print("=" * 55)
print("MASK QUALITY VERIFICATION")
print("=" * 55)

for image_path in images:
    mask_path = MASK_DIR / f"{image_path.stem}.png"

    if not mask_path.exists():
        print(f"FAIL: Missing mask - {image_path.name}")
        failed += 1
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"FAIL: Cannot read - {image_path.name}")
        failed += 1
        continue

    if image.shape[:2] != mask.shape[:2]:
        print(f"FAIL: Size mismatch - {image_path.name}")
        failed += 1
        continue

    unique_values = set(np.unique(mask))

    if not unique_values.issubset({0, 255}):
        print(f"FAIL: Invalid pixel values - {image_path.name}")
        failed += 1
        continue

    white_pixels = np.count_nonzero(mask == 255)

    if white_pixels == 0:
        print(f"FAIL: Empty mask - {image_path.name}")
        failed += 1
        continue

    if white_pixels == mask.size:
        print(f"FAIL: Entire mask is white - {image_path.name}")
        failed += 1
        continue

    passed += 1

print("\n" + "=" * 55)
print(f"Total images: {len(images)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print("=" * 55)

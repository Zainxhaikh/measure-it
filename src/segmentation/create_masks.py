
from pathlib import Path

import cv2
import numpy as np


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "dataset" / "images"
MASK_DIR = PROJECT_ROOT / "dataset" / "masks"

# Only the three missing images
SELECTED_IMAGES = [
    "IMG20261119153150.jpg",
    "IMG20261119153200_01.jpg",
    "IMG20261119153605.jpg",
]

MASK_DIR.mkdir(parents=True, exist_ok=True)


def create_mask(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path.name}")
        return False

    points = []
    display_image = image.copy()
    window_name = "Card Mask Annotation"

    def mouse_callback(event, x, y, flags, param):
        nonlocal display_image, points

        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

            cv2.circle(
                display_image,
                (x, y),
                12,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                display_image,
                str(len(points)),
                (x + 15, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            if len(points) > 1:
                cv2.line(
                    display_image,
                    points[-2],
                    points[-1],
                    (0, 255, 0),
                    3
                )

            if len(points) == 4:
                cv2.line(
                    display_image,
                    points[3],
                    points[0],
                    (0, 255, 0),
                    3
                )

            cv2.imshow(window_name, display_image)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 750)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\nImage: {image_path.name}")
    print("Click the four outer corners of the card:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print("\nS = Save | R = Reset | Q = Skip")

    while True:
        cv2.imshow(window_name, display_image)

        key = cv2.waitKey(20) & 0xFF

        # Accept both uppercase and lowercase keys
        if key != 255:
            key = ord(chr(key).lower())

        if key == ord("r"):
            points = []
            display_image = image.copy()
            print("Points reset. Select the four corners again.")

        elif key == ord("s"):
            if len(points) != 4:
                print("Select all four corners before saving.")
                continue

            polygon = np.array(points, dtype=np.int32)

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)

            mask_path = MASK_DIR / f"{image_path.stem}.png"

            if cv2.imwrite(str(mask_path), mask):
                print(f"Mask saved successfully: {mask_path.name}")
                cv2.destroyWindow(window_name)
                return True
            else:
                print("ERROR: Mask could not be saved.")

        elif key == ord("q"):
            print(f"Skipped: {image_path.name}")
            cv2.destroyWindow(window_name)
            return False


def main():
    print("=" * 50)
    print("MISSING CARD MASK ANNOTATION")
    print("=" * 50)

    completed = 0
    skipped = 0

    for filename in SELECTED_IMAGES:
        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / f"{image_path.stem}.png"

        if not image_path.exists():
            print(f"Image not found: {filename}")
            skipped += 1
            continue

        if mask_path.exists():
            print(f"Mask already exists: {filename}")
            completed += 1
            continue

        if create_mask(image_path):
            completed += 1
        else:
            skipped += 1

    print("\n" + "=" * 50)
    print("ANNOTATION SUMMARY")
    print("=" * 50)
    print(f"Completed: {completed}")
    print(f"Skipped / Not completed: {skipped}")
    print(f"Total selected images: {len(SELECTED_IMAGES)}")
    print(f"Masks directory: {MASK_DIR}")


if __name__ == "__main__":
    main()

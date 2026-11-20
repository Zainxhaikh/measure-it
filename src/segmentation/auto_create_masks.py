
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_DIR = Path("dataset/images")

# Test outputs are kept separate from final training masks.
OUTPUT_DIR = Path("results/auto_mask_test")

MASK_DIR = OUTPUT_DIR / "masks"
PREVIEW_DIR = OUTPUT_DIR / "previews"

TEST_LIMIT = 5

# Approximate playing-card width / height ratio.
EXPECTED_RATIO = 0.716

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def calculate_side_lengths(points):
    points = points.reshape(4, 2).astype(np.float32)

    lengths = []

    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        lengths.append(np.linalg.norm(p2 - p1))

    return lengths


def calculate_card_ratio(points):
    lengths = calculate_side_lengths(points)

    width = (lengths[0] + lengths[2]) / 2
    height = (lengths[1] + lengths[3]) / 2

    if width == 0 or height == 0:
        return 0

    return min(width, height) / max(width, height)


def check_angles(points):
    points = points.reshape(4, 2).astype(np.float32)

    for i in range(4):
        previous = points[(i - 1) % 4]
        current = points[i]
        following = points[(i + 1) % 4]

        v1 = previous - current
        v2 = following - current

        denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

        if denominator == 0:
            return False

        cosine = abs(np.dot(v1, v2) / denominator)

        # Reject extremely sharp or irregular corners.
        if cosine > 0.55:
            return False

    return True


# --------------------------------------------------
# CARD DETECTION
# --------------------------------------------------

def detect_card(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast under uneven lighting.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        40,
        120
    )

    kernel = np.ones((5, 5), np.uint8)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = image.shape[0] * image.shape[1]

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)
        area_ratio = area / image_area

        # Ignore very small and extremely large contours.
        if not 0.005 < area_ratio < 0.35:
            continue

        perimeter = cv2.arcLength(contour, True)

        for epsilon_factor in [0.015, 0.02, 0.03, 0.04]:

            approx = cv2.approxPolyDP(
                contour,
                epsilon_factor * perimeter,
                True
            )

            if len(approx) != 4:
                continue

            if not cv2.isContourConvex(approx):
                continue

            ratio = calculate_card_ratio(approx)

            # Exclude square-like objects such as the ArUco marker.
            if not 0.55 <= ratio <= 0.90:
                continue

            if not check_angles(approx):
                continue

            ratio_error = abs(ratio - EXPECTED_RATIO)

            # Rank candidates by card-like shape and size.
            score = area_ratio * np.exp(-3.0 * ratio_error)

            candidates.append(
                (score, area, approx.copy())
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best_score, best_area, best_polygon = candidates[0]

    return best_polygon


# --------------------------------------------------
# CREATE AND SAVE MASK
# --------------------------------------------------

def process_image(image_path):

    print(f"\nProcessing: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        print("FAILED: Image could not be read.")
        return False

    polygon = detect_card(image)

    if polygon is None:
        print("FAILED: No reliable card-shaped candidate found.")
        return False

    # Create binary mask.
    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [polygon],
        255
    )

    # Create visual preview.
    overlay = image.copy()

    cv2.polylines(
        overlay,
        [polygon],
        True,
        (0, 255, 0),
        12
    )

    # Resize preview for easier viewing.
    height, width = overlay.shape[:2]

    scale = min(1200 / width, 1200 / height, 1.0)

    preview = cv2.resize(
        overlay,
        (int(width * scale), int(height * scale))
    )

    # Save files.
    mask_path = MASK_DIR / f"{image_path.stem}.png"

    preview_path = PREVIEW_DIR / f"{image_path.stem}_preview.jpg"

    cv2.imwrite(str(mask_path), mask)

    cv2.imwrite(str(preview_path), preview)

    ratio = calculate_card_ratio(polygon)

    print("Detection completed.")
    print(f"Detected card ratio: {ratio:.3f}")
    print(f"Mask: {mask_path}")
    print(f"Preview: {preview_path}")

    return True


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print("No images found.")
        return

    test_images = images[:TEST_LIMIT]

    print("=" * 55)
    print("AUTOMATIC PLAYING CARD MASK GENERATION")
    print("=" * 55)

    print(f"Total dataset images: {len(images)}")
    print(f"Images in this test: {len(test_images)}")

    successful = 0
    failed = 0

    for image_path in test_images:

        if process_image(image_path):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 55)
    print("TEST SUMMARY")
    print("=" * 55)

    print(f"Successful detections: {successful}")
    print(f"Failed detections: {failed}")
    print(f"Total tested: {len(test_images)}")

    print(f"\nMasks: {MASK_DIR}")
    print(f"Previews: {PREVIEW_DIR}")

    print("\nIMPORTANT:")
    print("These are experimental masks.")
    print("Inspect every preview before using them for training.")


if __name__ == "__main__":
    main()

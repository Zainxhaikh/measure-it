
from pathlib import Path
import cv2

# Project paths
IMAGE_DIR = Path("calibration/images")

# ChArUco board configuration
BOARD_SIZE = (5, 7)
SQUARE_LENGTH = 25.0
MARKER_LENGTH = 18.0

DICTIONARY = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

board = cv2.aruco.CharucoBoard(
    BOARD_SIZE,
    SQUARE_LENGTH,
    MARKER_LENGTH,
    DICTIONARY
)

detector = cv2.aruco.CharucoDetector(board)

def main():
    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    print("OpenCV version:", cv2.__version__)
    print("Calibration images found:", len(image_paths))
    print("-" * 50)

    successful = 0
    failed = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read:", image_path.name)
            failed += 1
            continue

        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            detector.detectBoard(image)
        )

        corner_count = (
            0 if charuco_ids is None else len(charuco_ids)
        )

        marker_count = (
            0 if marker_ids is None else len(marker_ids)
        )

        if corner_count >= 6:
            successful += 1
            status = "DETECTED"
        else:
            failed += 1
            status = "INSUFFICIENT CORNERS"

        print(
            f"{image_path.name}: "
            f"{status} | "
            f"Markers: {marker_count} | "
            f"ChArUco corners: {corner_count}"
        )

    print("-" * 50)
    print("Successful images:", successful)
    print("Insufficient/failed images:", failed)
    print("Total images:", len(image_paths))


if __name__ == "__main__":
    main()

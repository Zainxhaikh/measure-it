
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "dataset" / "images"

CALIBRATION_FILE = (
    PROJECT_ROOT / "results" / "calibration" /
    "k3_fixed_zero" / "camera_calibration.npz"
)

TEST_IMAGES = [
    "IMG20261119153156.jpg",
    "IMG20261119153157.jpg",
    "IMG20261119153200.jpg",
]

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "measurement" /
    "final_marker_detection_test"
)

MARKER_ID = 10
MARKER_SIZE_MM = 30.0


def main():
    data = np.load(CALIBRATION_FILE)

    camera_matrix = data["camera_matrix"]
    distortion = data["distortion_coefficients"]

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_6X6_250
    )

    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(
        dictionary, parameters
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in TEST_IMAGES:

        image = cv2.imread(str(IMAGE_DIR / filename))

        if image is None:
            print("Could not read:", filename)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # First attempt: original grayscale image
        corners, ids, rejected = detector.detectMarkers(gray)

        method = "Original grayscale"

        # Fallback: Otsu thresholding
        if ids is None or MARKER_ID not in ids.flatten():

            _, processed = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            corners, ids, rejected = detector.detectMarkers(
                processed
            )

            method = "Otsu threshold"

        print("\nIMAGE:", filename)

        if ids is None or MARKER_ID not in ids.flatten():
            print("RESULT: Marker ID 10 not detected.")
            continue

        marker_index = list(ids.flatten()).index(MARKER_ID)

        marker_corners = corners[marker_index]

        # Calculate undistorted camera matrix
        height, width = image.shape[:2]

        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion,
            (width, height),
            0,
            (width, height)
        )

        # Transform detected corners into undistorted coordinates
        transformed = cv2.undistortPoints(
            marker_corners,
            camera_matrix,
            distortion,
            P=new_matrix
        )

        points = transformed.reshape(4, 2)

        side_lengths = [
            np.linalg.norm(points[i] - points[(i + 1) % 4])
            for i in range(4)
        ]

        average_side_pixels = float(np.mean(side_lengths))

        pixels_per_mm = average_side_pixels / MARKER_SIZE_MM

        print("Detection method:", method)
        print("Detected marker ID:", MARKER_ID)
        print("Average marker side:", round(average_side_pixels, 2), "pixels")
        print("Pixels per mm:", round(pixels_per_mm, 4))

        # Save visualization on original image
        output = image.copy()

        cv2.aruco.drawDetectedMarkers(
            output,
            [marker_corners],
            np.array([[MARKER_ID]])
        )

        cv2.imwrite(
            str(OUTPUT_DIR / filename),
            output
        )

    print("\nTest images saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()


from pathlib import Path

import cv2
import numpy as np


# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_DIR = Path("dataset/images")

# Selected calibration: portrait-only, k3 fixed at zero
CALIBRATION_FILE = Path(
    "results/calibration/k3_fixed_zero/camera_calibration.npz"
)

OUTPUT_DIR = Path(
    "results/calibration/k3_fixed_zero/validation"
)

# Two additional images for validation
TEST_IMAGES = [
    "IMG20261119153154.jpg",
    "IMG20261119153211_01.jpg",
]

# Alpha 0: prioritize valid image area and reduce black borders
ALPHA = 0.0


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    # Check calibration file
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_FILE}"
        )

    # Load calibration parameters
    calibration = np.load(CALIBRATION_FILE)

    camera_matrix = calibration["camera_matrix"]
    distortion = calibration["distortion_coefficients"]

    print("Calibration loaded successfully.")
    print("Calibration file:", CALIBRATION_FILE)

    # Create output directory automatically
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process both test images
    for filename in TEST_IMAGES:

        print("\n" + "=" * 50)
        print("Processing:", filename)

        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            print("ERROR: Image not found:", image_path)
            continue

        # Read image
        image = cv2.imread(str(image_path))

        if image is None:
            print("ERROR: Could not read image.")
            continue

        height, width = image.shape[:2]

        print("Image resolution:", width, "x", height)

        # Verify calibration resolution
        if (width, height) != (3072, 4096):
            print("SKIPPED: Unexpected image resolution.")
            continue

        # Calculate optimal camera matrix using Alpha 0
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion,
            (width, height),
            ALPHA,
            (width, height)
        )

        # Apply undistortion
        undistorted = cv2.undistort(
            image,
            camera_matrix,
            distortion,
            None,
            new_camera_matrix
        )

        # Create side-by-side comparison
        comparison = np.hstack((image, undistorted))

        # Create separate folder for each image
        image_output_dir = OUTPUT_DIR / Path(filename).stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        cv2.imwrite(
            str(image_output_dir / "original.jpg"),
            image
        )

        cv2.imwrite(
            str(image_output_dir / "undistorted.jpg"),
            undistorted
        )

        cv2.imwrite(
            str(image_output_dir / "comparison.jpg"),
            comparison
        )

        print("Alpha:", ALPHA)
        print("ROI:", roi)
        print("Saved results in:", image_output_dir)

    print("\n" + "=" * 50)
    print("VALIDATION TESTS COMPLETED.")


if __name__ == "__main__":
    main()

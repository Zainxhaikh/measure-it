
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_PATH = Path("dataset/images/IMG20261119153145.jpg")

CALIBRATION_PATH = Path(
    "results/calibration/k3_fixed_zero/camera_calibration.npz"
)

OUTPUT_DIR = Path(
    "results/calibration/k3_fixed_zero/undistortion_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load calibration parameters
# --------------------------------------------------

calibration = np.load(CALIBRATION_PATH)

camera_matrix = calibration["camera_matrix"]
distortion = calibration["distortion_coefficients"]

# --------------------------------------------------
# Load test image
# --------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

height, width = image.shape[:2]

if (width, height) != (
    int(calibration["image_width"]),
    int(calibration["image_height"])
):
    raise ValueError("Test image resolution does not match calibration.")

# --------------------------------------------------
# Undistort image
# --------------------------------------------------

undistorted = cv2.undistort(
    image,
    camera_matrix,
    distortion
)

# --------------------------------------------------
# Save results
# --------------------------------------------------

cv2.imwrite(str(OUTPUT_DIR / "original.jpg"), image)
cv2.imwrite(str(OUTPUT_DIR / "undistorted.jpg"), undistorted)

comparison = np.hstack((image, undistorted))

cv2.imwrite(
    str(OUTPUT_DIR / "comparison.jpg"),
    comparison
)

print("Undistortion test completed.")
print("\nImage resolution:", width, "x", height)
print("Distortion coefficients:", distortion.flatten())

print("\nSaved files:")
print(OUTPUT_DIR / "original.jpg")
print(OUTPUT_DIR / "undistorted.jpg")
print(OUTPUT_DIR / "comparison.jpg")

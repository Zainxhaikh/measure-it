
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
    "results/calibration/k3_fixed_zero/alpha_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load calibration parameters
# --------------------------------------------------

calibration = np.load(CALIBRATION_PATH)

camera_matrix = calibration["camera_matrix"]
distortion = calibration["distortion_coefficients"]

# --------------------------------------------------
# Load original image
# --------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

height, width = image.shape[:2]
image_size = (width, height)

# --------------------------------------------------
# Generate alpha = 0 undistorted image
# --------------------------------------------------

new_matrix_0, roi_0 = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    distortion,
    image_size,
    0,
    image_size
)

undistorted_0 = cv2.undistort(
    image,
    camera_matrix,
    distortion,
    None,
    new_matrix_0
)

# --------------------------------------------------
# Generate alpha = 1 undistorted image
# --------------------------------------------------

new_matrix_1, roi_1 = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    distortion,
    image_size,
    1,
    image_size
)

undistorted_1 = cv2.undistort(
    image,
    camera_matrix,
    distortion,
    None,
    new_matrix_1
)

# --------------------------------------------------
# Save images
# --------------------------------------------------

cv2.imwrite(str(OUTPUT_DIR / "alpha_0.jpg"), undistorted_0)
cv2.imwrite(str(OUTPUT_DIR / "alpha_1.jpg"), undistorted_1)

# Original | Alpha 0 | Alpha 1
comparison = np.hstack((
    image,
    undistorted_0,
    undistorted_1
))

cv2.imwrite(
    str(OUTPUT_DIR / "alpha_comparison.jpg"),
    comparison
)

# --------------------------------------------------
# Display information
# --------------------------------------------------

print("ALPHA COMPARISON COMPLETED")
print("=" * 45)

print("\nImage resolution:", width, "x", height)

print("\nAlpha 0 ROI:", roi_0)
print("Alpha 1 ROI:", roi_1)

print("\nSaved files:")
print(OUTPUT_DIR / "alpha_0.jpg")
print(OUTPUT_DIR / "alpha_1.jpg")
print(OUTPUT_DIR / "alpha_comparison.jpg")

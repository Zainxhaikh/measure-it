
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_DIR = Path("dataset/images")

CALIBRATION_FILE = Path(
    "results/calibration/portrait_only/camera_calibration.npz"
)

OUTPUT_DIR = Path(
    "results/calibration/portrait_only/undistortion_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the same image as the previous test.
IMAGE_NAME = "IMG20261119153145.jpg"

# --------------------------------------------------
# Load image and calibration
# --------------------------------------------------

image_path = IMAGE_DIR / IMAGE_NAME

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {image_path}"
    )

calibration = np.load(CALIBRATION_FILE)

camera_matrix = calibration["camera_matrix"]
distortion = calibration["distortion_coefficients"]

calibration_width = int(calibration["image_width"])
calibration_height = int(calibration["image_height"])

height, width = image.shape[:2]

print("Test image:", IMAGE_NAME)
print("Dataset resolution:", width, "x", height)
print("Calibration resolution:",
      calibration_width, "x", calibration_height)

if (width, height) != (
    calibration_width,
    calibration_height
):
    raise ValueError("Image and calibration resolutions do not match.")

# --------------------------------------------------
# Apply portrait-only calibration
# --------------------------------------------------

undistorted = cv2.undistort(
    image,
    camera_matrix,
    distortion
)

# --------------------------------------------------
# Save outputs separately
# --------------------------------------------------

cv2.imwrite(
    str(OUTPUT_DIR / "original.jpg"),
    image
)

cv2.imwrite(
    str(OUTPUT_DIR / "undistorted_portrait.jpg"),
    undistorted
)

comparison = np.hstack((image, undistorted))

cv2.imwrite(
    str(OUTPUT_DIR / "comparison_portrait.jpg"),
    comparison
)

print("\nPortrait-only undistortion completed.")

print("Saved files:")
print(OUTPUT_DIR / "original.jpg")
print(OUTPUT_DIR / "undistorted_portrait.jpg")
print(OUTPUT_DIR / "comparison_portrait.jpg")


from pathlib import Path
import json

import cv2
import numpy as np

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_DIR = Path("calibration/images")
OUTPUT_DIR = Path("results/calibration/k3_fixed_zero")

BOARD_SIZE = (5, 7)
SQUARE_LENGTH = 25.0
MARKER_LENGTH = 18.0

MIN_CORNERS = 6
MIN_VALID_IMAGES = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Initialize ChArUco board
# --------------------------------------------------

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

board = cv2.aruco.CharucoBoard(
    BOARD_SIZE,
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)

detector = cv2.aruco.CharucoDetector(board)
board_corners = board.getChessboardCorners()

object_points = []
image_points = []
used_images = []
skipped_images = []

image_size = None

# --------------------------------------------------
# Detect corners in portrait images
# --------------------------------------------------

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

print("OpenCV version:", cv2.__version__)
print("Images found:", len(image_paths))
print("\nUsing portrait images only.\n")

for image_path in image_paths:

    image = cv2.imread(str(image_path))

    if image is None:
        skipped_images.append(image_path.name)
        print("SKIPPED - Could not read:", image_path.name)
        continue

    height, width = image.shape[:2]

    if width != 3072 or height != 4096:
        skipped_images.append(image_path.name)
        print("SKIPPED - Not portrait:", image_path.name)
        continue

    image_size = (width, height)

    corners, ids, marker_corners, marker_ids = (
        detector.detectBoard(image)
    )

    if ids is None or len(ids) < MIN_CORNERS:
        skipped_images.append(image_path.name)
        print("SKIPPED - Insufficient corners:", image_path.name)
        continue

    ids = ids.flatten()

    obj = board_corners[ids].astype(np.float32)
    img = corners.reshape(-1, 2).astype(np.float32)

    object_points.append(obj)
    image_points.append(img)
    used_images.append(image_path.name)

    print(
        f"ACCEPTED: {image_path.name} "
        f"| Corners: {len(ids)}"
    )

# --------------------------------------------------
# Validate calibration images
# --------------------------------------------------

print("\nAccepted images:", len(used_images))
print("Skipped images:", len(skipped_images))

if len(used_images) < MIN_VALID_IMAGES:
    raise RuntimeError(
        f"At least {MIN_VALID_IMAGES} valid images are required."
    )

# --------------------------------------------------
# Calibrate camera with k3 fixed at zero
# --------------------------------------------------

print("\nCalculating calibration with k3 fixed at zero...")

rms, camera_matrix, distortion, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3
    )
)

# --------------------------------------------------
# Calculate reprojection error
# --------------------------------------------------

all_errors = []
per_image_errors = []

for i in range(len(object_points)):

    projected, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion
    )

    projected = projected.reshape(-1, 2)
    actual = image_points[i].reshape(-1, 2)

    distances = np.linalg.norm(actual - projected, axis=1)

    image_error = float(np.mean(distances))
    all_errors.extend(distances.tolist())

    per_image_errors.append({
        "image": used_images[i],
        "mean_error_pixels": image_error
    })

mean_error = float(np.mean(all_errors))

# --------------------------------------------------
# Save calibration parameters
# --------------------------------------------------

np.savez(
    OUTPUT_DIR / "camera_calibration.npz",
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion,
    image_width=image_size[0],
    image_height=image_size[1],
    rms_error=rms
)

# --------------------------------------------------
# Save calibration report
# --------------------------------------------------

report = {
    "experiment": "Portrait-only calibration with k3 fixed at zero",
    "calibration_flags": ["cv2.CALIB_FIX_K3"],
    "opencv_version": cv2.__version__,
    "image_resolution": {
        "width": image_size[0],
        "height": image_size[1]
    },
    "board_configuration": {
        "squares_x": BOARD_SIZE[0],
        "squares_y": BOARD_SIZE[1],
        "square_length_mm": SQUARE_LENGTH,
        "marker_length_mm": MARKER_LENGTH
    },
    "total_images": len(image_paths),
    "accepted_images": used_images,
    "skipped_images": skipped_images,
    "accepted_count": len(used_images),
    "skipped_count": len(skipped_images),
    "camera_matrix": camera_matrix.tolist(),
    "distortion_coefficients": distortion.flatten().tolist(),
    "fixed_k3_value": 0.0,
    "opencv_rms_reprojection_error_pixels": float(rms),
    "mean_point_reprojection_error_pixels": mean_error,
    "per_image_errors": per_image_errors
}

with open(
    OUTPUT_DIR / "calibration_report.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(report, file, indent=4)

# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 50)
print("K3 FIXED AT ZERO CALIBRATION COMPLETED")
print("=" * 50)

print("\nCamera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(distortion)

print(f"\nRMS reprojection error: {rms:.4f} pixels")
print(f"Mean point reprojection error: {mean_error:.4f} pixels")

print("\nAccepted images:", len(used_images))
print("Skipped images:", len(skipped_images))

print("\nSaved files:")
print(OUTPUT_DIR / "camera_calibration.npz")
print(OUTPUT_DIR / "calibration_report.json")

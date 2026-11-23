
from pathlib import Path
import json

import cv2
import numpy as np

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_DIR = Path("calibration/images")

# Save this experiment separately.
OUTPUT_DIR = Path("results/calibration/k3_fixed_zero")

BOARD_SIZE = (5, 7)
SQUARE_LENGTH = 25.0
MARKER_LENGTH = 18.0

MIN_CORNERS = 6
MIN_VALID_IMAGES = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Initialize ChArUco board and detector
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
# Detect ChArUco corners in calibration images
# --------------------------------------------------

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

print("OpenCV version:", cv2.__version__)
print("Images found:", len(image_paths))
print("\nDetecting calibration points...\n")

for image_path in image_paths:

    image = cv2.imread(str(image_path))

    if image is None:
        skipped_images.append(image_path.name)
        print("SKIPPED - Could not read:", image_path.name)
        continue

    # Use portrait images only for this experiment.
    # Landscape images are excluded rather than rotated.
    if image.shape[1] > image.shape[0]:
        skipped_images.append(image_path.name)
        print("SKIPPED - Landscape image:", image_path.name)
        continue

    current_size = (image.shape[1], image.shape[0])

    if image_size is None:
        image_size = current_size

    elif current_size != image_size:
        skipped_images.append(image_path.name)
        print("SKIPPED - Different image size:", image_path.name)
        continue

    # Detect ChArUco board.
    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        detector.detectBoard(image)
    )

    if charuco_ids is None or len(charuco_ids) < MIN_CORNERS:
        skipped_images.append(image_path.name)
        print("SKIPPED - Insufficient corners:", image_path.name)
        continue

    ids = charuco_ids.flatten()

    # Match image points with known board coordinates.
    obj = board_corners[ids].astype(np.float32)
    img = charuco_corners.reshape(-1, 2).astype(np.float32)

    object_points.append(obj)
    image_points.append(img)
    used_images.append(image_path.name)

    print(
        f"ACCEPTED: {image_path.name} "
        f"| Corners: {len(ids)}"
    )

# --------------------------------------------------
# Check minimum calibration requirements
# --------------------------------------------------

print("\nAccepted images:", len(used_images))
print("Skipped images:", len(skipped_images))

if len(used_images) < MIN_VALID_IMAGES:
    raise RuntimeError(
        f"Need at least {MIN_VALID_IMAGES} valid images. "
        f"Only {len(used_images)} were detected."
    )

# --------------------------------------------------
# Calculate camera calibration with k3 fixed at zero
# --------------------------------------------------

print("\nCalculating camera calibration...")
print("Experiment: Fixing the k3 distortion coefficient at zero.")

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
# Calculate reprojection errors
# --------------------------------------------------

per_image_errors = []
all_point_errors = []

for i in range(len(object_points)):

    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion
    )

    projected_points = projected_points.reshape(-1, 2)
    actual_points = image_points[i].reshape(-1, 2)

    distances = np.linalg.norm(
        actual_points - projected_points,
        axis=1
    )

    mean_error = float(np.mean(distances))

    per_image_errors.append({
        "image": used_images[i],
        "mean_error_pixels": mean_error,
        "corner_count": len(actual_points)
    })

    all_point_errors.extend(distances.tolist())

mean_reprojection_error = float(
    np.mean(all_point_errors)
)

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
    "fixed_distortion_coefficient": "k3",
    "fixed_k3_value": 0.0,
    "calibration_flags": ["cv2.CALIB_FIX_K3"],
    "opencv_version": cv2.__version__,
    "image_resolution": {
        "width": image_size[0],
        "height": image_size[1]
    },
    "orientation_normalization": "Portrait images only; landscape images excluded",
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
    "opencv_rms_reprojection_error_pixels": float(rms),
    "mean_point_reprojection_error_pixels": mean_reprojection_error,
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

print("\n" + "=" * 55)
print("K3 FIXED AT ZERO - CALIBRATION COMPLETED")
print("=" * 55)

print("\nCamera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(distortion)

print(f"\nOpenCV RMS reprojection error: {rms:.4f} pixels")

print(
    "Mean point reprojection error: "
    f"{mean_reprojection_error:.4f} pixels"
)

print("\nAccepted images:", len(used_images))
print("Skipped images:", len(skipped_images))

print("\nSaved files:")
print(OUTPUT_DIR / "camera_calibration.npz")
print(OUTPUT_DIR / "calibration_report.json")

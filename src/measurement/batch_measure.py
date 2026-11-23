
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.model import UNetResNet18
from src.measurement.config import (
    CALIBRATION_FILE,
    MODEL_WEIGHTS,
    CALIBRATION_WIDTH,
    CALIBRATION_HEIGHT,
    UNDISTORTION_ALPHA,
    ARUCO_MARKER_ID,
    ARUCO_MARKER_SIZE_MM,
    IMAGE_SIZE,
    SEGMENTATION_THRESHOLD,
    MEASUREMENT_RESULTS_DIR,
)

IMAGE_DIR = PROJECT_ROOT / "dataset" / "images"
DEVICE = torch.device("cpu")


def load_calibration():
    data = np.load(CALIBRATION_FILE)
    camera_matrix = data["camera_matrix"]
    distortion = data["distortion_coefficients"]

    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (CALIBRATION_WIDTH, CALIBRATION_HEIGHT),
        UNDISTORTION_ALPHA,
        (CALIBRATION_WIDTH, CALIBRATION_HEIGHT),
    )

    return camera_matrix, distortion, new_matrix


def load_model():
    model = UNetResNet18(pretrained=False).to(DEVICE)
    state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def detect_reference_marker(image, camera_matrix, distortion, new_matrix):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_6X6_250
    )
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Try original grayscale first, then Otsu threshold.
    attempts = [
        ("Original grayscale", gray),
        ("Otsu threshold", cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]),
    ]

    for method, processed in attempts:
        corners, ids, _ = detector.detectMarkers(processed)

        if ids is None:
            continue

        for index, marker_id in enumerate(ids.flatten()):
            if int(marker_id) != ARUCO_MARKER_ID:
                continue

            original_corners = corners[index].reshape(1, 4, 2)

            transformed = cv2.undistortPoints(
                original_corners,
                camera_matrix,
                distortion,
                P=new_matrix,
            ).reshape(4, 2)

            side_lengths = [
                np.linalg.norm(transformed[i] - transformed[(i + 1) % 4])
                for i in range(4)
            ]

            marker_pixels = float(np.mean(side_lengths))

            if marker_pixels <= 0:
                return None

            return transformed, marker_pixels, method

    return None


def segment_card(image, model):
    resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(
        rgb.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probability = torch.sigmoid(output)[0, 0].cpu().numpy()

    mask = (probability > SEGMENTATION_THRESHOLD).astype(np.uint8)

    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    confidence = float(probability[probability > SEGMENTATION_THRESHOLD].mean()) \
        if np.any(probability > SEGMENTATION_THRESHOLD) else 0.0

    return mask, confidence


def measure_card(mask, pixels_per_mm):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    rectangle = cv2.minAreaRect(contour)

    width_pixels, height_pixels = sorted(rectangle[1])

    if width_pixels <= 0 or height_pixels <= 0:
        return None

    return (
        width_pixels / pixels_per_mm,
        height_pixels / pixels_per_mm,
        contour,
    )


def process_image(image_path, model, calibration):
    camera_matrix, distortion, new_matrix = calibration

    image = cv2.imread(str(image_path))
    if image is None:
        return None, "Could not read image"

    if image.shape[1] != CALIBRATION_WIDTH or image.shape[0] != CALIBRATION_HEIGHT:
        return None, "Unexpected image resolution"

    marker_result = detect_reference_marker(
        image, camera_matrix, distortion, new_matrix
    )

    if marker_result is None:
        return None, "ArUco marker ID 10 not detected"

    marker_corners, marker_pixels, method = marker_result
    pixels_per_mm = marker_pixels / ARUCO_MARKER_SIZE_MM

    undistorted = cv2.undistort(
        image, camera_matrix, distortion, None, new_matrix
    )

    mask, confidence = segment_card(undistorted, model)
    measurement = measure_card(mask, pixels_per_mm)

    if measurement is None:
        return None, "Card segmentation or measurement failed"

    width_mm, height_mm, contour = measurement

    overlay = undistorted.copy()
    overlay[mask > 0] = (
        0.5 * overlay[mask > 0] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 5)
    cv2.polylines(
        overlay,
        [np.round(marker_corners).astype(np.int32)],
        True,
        (255, 0, 0),
        5,
    )

    MEASUREMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        MEASUREMENT_RESULTS_DIR /
        f"{image_path.stem}_measurement.jpg"
    )
    cv2.imwrite(str(output_path), overlay)

    return {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "confidence": confidence,
        "pixels_per_mm": pixels_per_mm,
        "marker_method": method,
        "overlay": output_path,
    }, None


def main():
    print("=" * 55)
    print("BATCH CARD MEASUREMENT")
    print("=" * 55)

    calibration = load_calibration()
    model = load_model()

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))[:10]

    successful = 0
    failed = 0

    for image_path in image_paths:
        print(f"\nImage: {image_path.name}")

        result, error = process_image(
            image_path, model, calibration
        )

        if error:
            failed += 1
            print(f"FAILED: {error}")
            continue

        successful += 1
        print(f"Marker method: {result['marker_method']}")
        print(f"Pixels per mm: {result['pixels_per_mm']:.4f}")
        print(f"Width: {result['width_mm']:.2f} mm")
        print(f"Height: {result['height_mm']:.2f} mm")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Overlay: {result['overlay']}")

    print("\n" + "=" * 55)
    print(f"SUCCESSFUL: {successful}/{len(image_paths)}")
    print(f"FAILED: {failed}/{len(image_paths)}")
    print("=" * 55)


if __name__ == "__main__":
    main()

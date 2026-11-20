
from pathlib import Path
import argparse
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "src" / "segmentation"))

from model import UNetResNet18

from reliability_checks import check_measurement_reliability


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CALIBRATION_FILE = (
    PROJECT_ROOT / "results" / "calibration" /
    "k3_fixed_zero" / "camera_calibration.npz"
)

MODEL_FILE = PROJECT_ROOT / "weights" / "best_card_segmentation.pth"

OUTPUT_DIR = PROJECT_ROOT / "results" / "measurement"

MARKER_ID = 10
MARKER_SIZE_MM = 30.0

IMAGE_SIZE = 512
THRESHOLD = 0.5

DEVICE = torch.device("cpu")


# --------------------------------------------------
# LOAD CALIBRATION
# --------------------------------------------------

def load_calibration():

    data = np.load(CALIBRATION_FILE)

    camera_matrix = data["camera_matrix"]
    distortion = data["distortion_coefficients"]

    return camera_matrix, distortion


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

def load_model():

    model = UNetResNet18(pretrained=False).to(DEVICE)

    weights = torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(weights)
    model.eval()

    return model


# --------------------------------------------------
# ESTIMATE MARKER PLANE TILT
# --------------------------------------------------

def estimate_marker_tilt(
    original_corners,
    camera_matrix,
    distortion
):

    half_size = MARKER_SIZE_MM / 2.0

    object_points = np.array([
        [-half_size, half_size, 0],
        [half_size, half_size, 0],
        [half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)

    image_points = original_corners.reshape(4, 2).astype(
        np.float32
    )

    success, rotation_vector, translation_vector = (
        cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    )

    if not success:
        return float("nan")

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    normal = rotation_matrix[:, 2]

    cosine_angle = np.clip(
        abs(normal[2]),
        0.0,
        1.0
    )

    tilt_angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return float(tilt_angle)


# --------------------------------------------------
# DETECT REFERENCE MARKER
# --------------------------------------------------

def detect_reference_marker(
    image,
    camera_matrix,
    distortion,
    new_matrix
):

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_6X6_250
    )

    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(
        dictionary,
        parameters
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    corners, ids, _ = detector.detectMarkers(gray)

    detection_method = "Original grayscale"

    # Fallback: Otsu thresholding
    if ids is None or MARKER_ID not in ids.flatten():

        _, processed = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        corners, ids, _ = detector.detectMarkers(processed)

        detection_method = "Otsu threshold"

    # Missing marker
    if ids is None or MARKER_ID not in ids.flatten():

        return None, None, 0.0, 0, float("nan"), detection_method

    marker_index = list(ids.flatten()).index(MARKER_ID)

    original_corners = corners[marker_index]

    corner_count = len(original_corners.reshape(-1, 2))

    if corner_count != 4:

        return (
            None,
            None,
            0.0,
            corner_count,
            float("nan"),
            detection_method
        )

    # Transform marker corners into undistorted coordinates
    undistorted_corners = cv2.undistortPoints(
        original_corners,
        camera_matrix,
        distortion,
        P=new_matrix
    )

    points = undistorted_corners.reshape(4, 2)

    side_lengths = [
        np.linalg.norm(
            points[i] - points[(i + 1) % 4]
        )
        for i in range(4)
    ]

    marker_pixels = float(np.mean(side_lengths))

    tilt_angle = estimate_marker_tilt(
        original_corners,
        camera_matrix,
        distortion
    )

    print("Marker detection method:", detection_method)

    return (
        points,
        original_corners,
        marker_pixels,
        corner_count,
        tilt_angle,
        detection_method
    )


# --------------------------------------------------
# CARD SEGMENTATION
# --------------------------------------------------

def predict_card_mask(model, image):

    height, width = image.shape[:2]

    resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    tensor = torch.from_numpy(
        rgb.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():

        output = model(tensor.to(DEVICE))

        probabilities = (
            torch.sigmoid(output)[0, 0].cpu().numpy()
        )

    small_mask = (
        probabilities > THRESHOLD
    ).astype(np.uint8) * 255

    full_mask = cv2.resize(
        small_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    foreground = probabilities > THRESHOLD

    confidence = (
        float(probabilities[foreground].mean())
        if np.any(foreground)
        else 0.0
    )

    return full_mask, confidence


# --------------------------------------------------
# MEASURE CARD
# --------------------------------------------------

def measure_card(mask, pixels_per_mm):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError(
            "Card segmentation failed: no card detected."
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    card_area = float(
        cv2.contourArea(largest_contour)
    )

    if card_area < 1000:
        raise ValueError(
            "Detected card region is too small."
        )

    rectangle = cv2.minAreaRect(largest_contour)

    box = cv2.boxPoints(rectangle)
    box = np.int32(box)

    width_pixels, height_pixels = sorted(
        rectangle[1]
    )

    if width_pixels <= 0 or height_pixels <= 0:
        raise ValueError("Invalid card dimensions.")

    width_mm = width_pixels / pixels_per_mm
    height_mm = height_pixels / pixels_per_mm

    return width_mm, height_mm, box, card_area


# --------------------------------------------------
# RELIABILITY VALIDATION
# --------------------------------------------------

def validate_measurement(
    confidence,
    marker_detected,
    marker_corner_count,
    card_area,
    image_area,
    pixels_per_mm,
    tilt_angle
):

    result = check_measurement_reliability(
        confidence=confidence,
        marker_detected=marker_detected,
        marker_corner_count=marker_corner_count,
        card_area=card_area,
        image_area=image_area,
        pixels_per_mm=pixels_per_mm,
        tilt_angle=tilt_angle
    )

    print("\nRELIABILITY CHECK")
    print("=================")
    print("Reliable:", result["reliable"])

    if result["reasons"]:

        for reason in result["reasons"]:
            print("-", reason)

    return result["reliable"], result["reasons"]


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Measure a playing card using image "
            "segmentation and an ArUco reference marker."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the input image"
    )

    args = parser.parse_args()

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            "Camera calibration file not found."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained segmentation model not found."
        )

    image_path = Path(args.image)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    camera_matrix, distortion = load_calibration()

    height, width = image.shape[:2]

    if (width, height) != (3072, 4096):
        raise ValueError(
            "Input image must have calibrated portrait "
            "resolution 3072x4096."
        )

    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (width, height),
        0,
        (width, height)
    )

    # Detect reference marker
    (
        marker_corners,
        original_marker_corners,
        marker_pixels,
        marker_corner_count,
        tilt_angle,
        detection_method
    ) = detect_reference_marker(
        image,
        camera_matrix,
        distortion,
        new_matrix
    )

    marker_detected = marker_corners is not None

    pixels_per_mm = (
        marker_pixels / MARKER_SIZE_MM
        if marker_detected
        else 0.0
    )

    # Reject missing or incomplete marker before measurement
    if not marker_detected:

        reliable, reasons = validate_measurement(
            confidence=0.0,
            marker_detected=False,
            marker_corner_count=marker_corner_count,
            card_area=0.0,
            image_area=width * height,
            pixels_per_mm=0.0,
            tilt_angle=tilt_angle
        )

        raise ValueError(
            "Measurement rejected: " + " ".join(reasons)
        )

    # Undistort the complete image
    undistorted = cv2.undistort(
        image,
        camera_matrix,
        distortion,
        None,
        new_matrix
    )

    # Load model and predict card mask
    model = load_model()

    card_mask, confidence = predict_card_mask(
        model,
        undistorted
    )

    # Get card area for reliability validation
    card_area = float(
        np.count_nonzero(card_mask)
    )

    image_area = float(width * height)

    reliable, reasons = validate_measurement(
        confidence=confidence,
        marker_detected=marker_detected,
        marker_corner_count=marker_corner_count,
        card_area=card_area,
        image_area=image_area,
        pixels_per_mm=pixels_per_mm,
        tilt_angle=tilt_angle
    )

    if not reliable:

        raise ValueError(
            "Measurement rejected: " + " ".join(reasons)
        )

    # Calculate card dimensions
    card_width, card_height, box, contour_area = measure_card(
        card_mask,
        pixels_per_mm
    )

    # --------------------------------------------------
    # SAVE MEASUREMENT OVERLAY
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    overlay = undistorted.copy()

    green = np.array(
        [0, 255, 0],
        dtype=np.float32
    )

    overlay[card_mask > 0] = (
        0.5 * overlay[card_mask > 0].astype(np.float32)
        + 0.5 * green
    ).astype(np.uint8)

    cv2.drawContours(
        overlay,
        [box],
        -1,
        (0, 0, 255),
        8
    )

    marker_box = np.int32(
        np.round(marker_corners)
    )

    cv2.polylines(
        overlay,
        [marker_box],
        True,
        (255, 0, 0),
        8
    )

    cv2.putText(
        overlay,
        f"Width: {card_width:.2f} mm",
        (100, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        4
    )

    cv2.putText(
        overlay,
        f"Height: {card_height:.2f} mm",
        (100, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        4
    )

    output_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_measurement.jpg"
    )

    cv2.imwrite(
        str(output_path),
        overlay
    )

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    print("\nCARD MEASUREMENT")
    print("================")

    print(f"Image: {image_path.name}")
    print(f"Reference marker: ID {MARKER_ID}")
    print(f"Marker size: {MARKER_SIZE_MM:.2f} mm")
    print(f"Marker side: {marker_pixels:.2f} pixels")
    print(f"Pixels per mm: {pixels_per_mm:.4f}")
    print(f"Segmentation confidence: {confidence:.4f}")
    print(f"Estimated marker plane tilt: {tilt_angle:.2f} degrees")

    print(f"Card width: {card_width:.2f} mm")
    print(f"Card height: {card_height:.2f} mm")

    print("Reliability status: ACCEPTED")
    print(f"Overlay saved: {output_path}")


if __name__ == "__main__":
    main()

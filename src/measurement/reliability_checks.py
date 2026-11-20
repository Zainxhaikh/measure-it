
import math


def check_measurement_reliability(
    confidence,
    marker_detected,
    marker_corner_count,
    card_area,
    image_area,
    pixels_per_mm,
    tilt_angle=0.0
):
    reasons = []

    # Check reference marker
    if not marker_detected:
        reasons.append("Reference marker not detected.")

    if marker_corner_count != 4:
        reasons.append("Reference marker is incomplete.")

    # Check segmentation confidence
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        reasons.append("Invalid segmentation confidence.")
    elif confidence < 0.70:
        reasons.append("Segmentation confidence is too low.")

    # Check detected card area
    if (
        not math.isfinite(card_area)
        or not math.isfinite(image_area)
        or image_area <= 0
        or card_area <= 0
    ):
        reasons.append("Invalid card or image area.")

    elif card_area / image_area < 0.01:
        reasons.append("Detected card region is too small.")

    # Check pixel-to-mm scale
    if (
        not math.isfinite(pixels_per_mm)
        or pixels_per_mm <= 0
    ):
        reasons.append("Invalid pixel-to-mm scale.")

    # Check card tilt
    if (
        not math.isfinite(tilt_angle)
        or abs(tilt_angle) > 15
    ):
        reasons.append("Card tilt is too high.")

    reliable = len(reasons) == 0

    return {
        "reliable": reliable,
        "reasons": reasons
    }


if __name__ == "__main__":

    result = check_measurement_reliability(
        confidence=0.90,
        marker_detected=True,
        marker_corner_count=4,
        card_area=500000,
        image_area=3072 * 4096,
        pixels_per_mm=10.0,
        tilt_angle=5.0
    )

    print("RELIABILITY CHECK")
    print("=================")
    print("Reliable:", result["reliable"])

    if result["reasons"]:
        for reason in result["reasons"]:
            print("-", reason)
    else:
        print("No reliability issues detected.")


import math

from reliability_checks import check_measurement_reliability


def main():

    valid_data = {
        "confidence": 0.90,
        "marker_detected": True,
        "marker_corner_count": 4,
        "card_area": 500000,
        "image_area": 3072 * 4096,
        "pixels_per_mm": 10.0,
        "tilt_angle": 5.0,
    }

    tests = [
        ("Valid measurement", {}, True),

        ("Missing marker", {
            "marker_detected": False,
            "marker_corner_count": 0
        }, False),

        ("Incomplete marker", {
            "marker_corner_count": 3
        }, False),

        ("Low segmentation confidence", {
            "confidence": 0.40
        }, False),

        ("Invalid confidence", {
            "confidence": 1.5
        }, False),

        ("High card tilt", {
            "tilt_angle": 25.0
        }, False),

        ("Invalid scale", {
            "pixels_per_mm": 0.0
        }, False),

        ("NaN scale", {
            "pixels_per_mm": math.nan
        }, False),

        ("Invalid image area", {
            "image_area": 0
        }, False),

        ("Card area too small", {
            "card_area": 100
        }, False),

        ("Invalid tilt value", {
            "tilt_angle": math.nan
        }, False),
    ]

    passed = 0

    print("RELIABILITY UNIT TESTS")
    print("======================")

    for name, overrides, expected in tests:

        test_data = valid_data.copy()
        test_data.update(overrides)

        result = check_measurement_reliability(**test_data)

        success = result["reliable"] == expected

        print(f"{name}: {'PASS' if success else 'FAIL'}")

        if not success:
            print("  Expected:", expected)
            print("  Actual:", result["reliable"])
            print("  Reasons:", result["reasons"])

        passed += int(success)

    print(f"\nTests passed: {passed}/{len(tests)}")

    if passed != len(tests):
        raise AssertionError("Some reliability tests failed.")


if __name__ == "__main__":
    main()

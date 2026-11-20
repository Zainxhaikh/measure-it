
from reliability_checks import check_measurement_reliability


def run_adverse_tests():
    tests = [
        {
            "name": "Normal lighting and low tilt",
            "confidence": 0.92,
            "marker_detected": True,
            "marker_corner_count": 4,
            "card_area": 500000,
            "image_area": 3072 * 4096,
            "pixels_per_mm": 10.0,
            "tilt_angle": 5.0,
            "expected_reliable": True,
        },
        {
            "name": "Low lighting - low confidence",
            "confidence": 0.45,
            "marker_detected": True,
            "marker_corner_count": 4,
            "card_area": 500000,
            "image_area": 3072 * 4096,
            "pixels_per_mm": 10.0,
            "tilt_angle": 5.0,
            "expected_reliable": False,
        },
        {
            "name": "Partial reference marker",
            "confidence": 0.90,
            "marker_detected": True,
            "marker_corner_count": 2,
            "card_area": 500000,
            "image_area": 3072 * 4096,
            "pixels_per_mm": 10.0,
            "tilt_angle": 5.0,
            "expected_reliable": False,
        },
        {
            "name": "High camera tilt",
            "confidence": 0.90,
            "marker_detected": True,
            "marker_corner_count": 4,
            "card_area": 500000,
            "image_area": 3072 * 4096,
            "pixels_per_mm": 10.0,
            "tilt_angle": 25.0,
            "expected_reliable": False,
        },
        {
            "name": "Very small detected card",
            "confidence": 0.90,
            "marker_detected": True,
            "marker_corner_count": 4,
            "card_area": 1000,
            "image_area": 3072 * 4096,
            "pixels_per_mm": 10.0,
            "tilt_angle": 5.0,
            "expected_reliable": False,
        },
    ]

    print("ADVERSE CONDITION TEST PLAN")
    print("===========================")

    for test in tests:
        print(f"\nTest: {test['name']}")
        print(f"Expected reliable: {test['expected_reliable']}")

    print("\nAdverse-condition test cases prepared.")
    print("Actual image-based testing will be performed later.")


if __name__ == "__main__":
    run_adverse_tests()

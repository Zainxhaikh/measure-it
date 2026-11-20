
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

    passed = 0

    print("ADVERSE CONDITION RELIABILITY TESTS")
    print("===================================")

    for test in tests:
        expected = test["expected_reliable"]

        result = check_measurement_reliability(
            confidence=test["confidence"],
            marker_detected=test["marker_detected"],
            marker_corner_count=test["marker_corner_count"],
            card_area=test["card_area"],
            image_area=test["image_area"],
            pixels_per_mm=test["pixels_per_mm"],
            tilt_angle=test["tilt_angle"],
        )

        success = result["reliable"] == expected

        if success:
            passed += 1

        print(f"\n{test['name']}: {'PASS' if success else 'FAIL'}")
        print(f"Expected reliable: {expected}")
        print(f"Actual reliable: {result['reliable']}")

        for reason in result["reasons"]:
            print(f"- {reason}")

    print("\n===================================")
    print(f"Tests passed: {passed}/{len(tests)}")

    if passed != len(tests):
        raise SystemExit("Some adverse-condition tests failed.")


if __name__ == "__main__":
    run_adverse_tests()

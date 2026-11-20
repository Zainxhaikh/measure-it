# Testing and Validation Report

## 1. Camera Calibration

- Total calibration images: 45
- Accepted images: 35
- Skipped images: 10
- OpenCV RMS reprojection error: 2.1247 pixels
- Mean point reprojection error: 1.4205 pixels
- Selected configuration: Portrait-only calibration with k3 fixed at zero
- Undistortion alpha: 0.0

## 2. Dataset Verification

- Total images: 83
- Training images: 58
- Validation images: 16
- Testing images: 9
- Verified masks: 83
- Failed mask checks: 0

## 3. Segmentation Model Results

To be completed after training and testing.

- Training epochs:
- Best validation IoU:
- Test IoU:
- mAP@0.5:
- mAP@0.5:0.95:
- Precision:
- Recall:

## 4. Measurement Accuracy

To be completed using at least 10 genuine manual measurements.

- Number of comparisons:
- Mean Absolute Error (MAE):
- Mean percentage error:
- Mean error:
- Median error:
- Worst-case error:

## 5. Reliability Testing

Record actual test results for:

- Missing reference marker
- Partial reference marker
- Low segmentation confidence
- High camera tilt
- Small or partially visible card
- Different lighting conditions

## 6. Undistortion Comparison

Record the observed difference between original and undistorted images.

## 7. Additional Measurable Objective

To be completed after selecting and evaluating the second measurable objective.

## 8. Final Conclusion

To be completed after all experiments and validation are finished.
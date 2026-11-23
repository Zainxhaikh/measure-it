
# Testing and Validation Report

## 1. Camera Calibration

The camera calibration results are summarized below:

| Parameter | Result |
|---|---:|
| Total calibration images | 45 |
| Accepted images | 35 |
| Skipped images | 10 |
| OpenCV RMS reprojection error | 2.1247 pixels |
| Mean point reprojection error | 1.4205 pixels |
| Selected configuration | Portrait-only calibration with k3 fixed at zero |
| Undistortion alpha | 0.0 |

The selected calibration configuration is used for image undistortion before segmentation and measurement.

## 2. Dataset Verification

The dataset was checked to verify image and mask availability and the dataset split.

| Parameter | Result |
|---|---:|
| Total images | 83 |
| Training images | 58 |
| Validation images | 16 |
| Testing images | 9 |
| Verified masks | 83 |
| Failed mask checks | 0 |

All 83 masks passed the reported verification checks.

## 3. Segmentation Model Results

The following metrics are reserved for reporting segmentation model performance.

- Training epochs: To be documented from training results.
- Best validation IoU: To be documented.
- Test IoU: To be documented.
- mAP@0.5: To be documented.
- mAP@0.5:0.95: To be documented.
- Precision: To be documented.
- Recall: To be documented.

These values should be taken from the actual model evaluation output.

## 4. Measurement Accuracy

The measurement comparison was performed using 10 images. The physical playing card dimensions were measured as 57 mm in width and 86 mm in height using a ruler. The measurements were checked multiple times.

The ArUco reference marker size was confirmed as 30 mm across its outer black boundary.

### Width Measurement Results

| Metric | Result |
|---|---:|
| Number of comparisons | 10 |
| Mean Absolute Error (MAE) | 3.7460 mm |
| Mean Error | -3.7460 mm |
| Median Absolute Error | 3.7400 mm |
| Worst-case Error | 4.1600 mm |
| Mean Percentage Error | 6.5719% |

### Height Measurement Results

| Metric | Result |
|---|---:|
| Number of comparisons | 10 |
| Mean Absolute Error (MAE) | 0.9560 mm |
| Mean Error | -0.9560 mm |
| Median Absolute Error | 0.8950 mm |
| Worst-case Error | 1.3100 mm |
| Mean Percentage Error | 1.1116% |

### Measurement Observation

The AI system consistently produced measurements below the physical reference dimensions in the evaluated images.

The width measurements showed a larger error than the height measurements. This difference is documented for further investigation and improvement.

The comparison results are available in:

`results/measurement/measurement_comparison_report.txt`

## 5. Reliability Testing

Five reliability scenarios were executed using simulated inputs through the `check_measurement_reliability()` function.

| Test Scenario | Expected Result | Actual Result |
|---|---|---|
| Normal lighting and low tilt | Reliable | PASS |
| Low confidence (simulated poor lighting) | Rejected | PASS |
| Partial reference marker (simulated incomplete marker) | Rejected | PASS |
| High camera tilt | Rejected | PASS |
| Very small detected card | Rejected | PASS |

**Result:** 5 out of 5 simulated reliability tests passed.

**Limitation:** These tests validate the reliability-checking logic using controlled inputs. They do not represent physical camera tests under actual poor lighting, partial marker occlusion, or high-tilt conditions. Further real-world validation is required.

## 6. Undistortion Comparison

Visual inspection was performed on three dataset images after applying the selected camera calibration.

The observations included:

- Card edges remained visually straight without additional bending.
- The ArUco marker retained its square appearance.
- Background edges remained visually flat.
- No significant black borders or unwanted stretching were observed.

These observations provide qualitative validation of the undistortion process. A separate quantitative comparison of measurement results before and after undistortion is still required.

## 7. Additional Measurable Objective

The second measurable objective is reserved for evaluation and documentation after the relevant experiment has been completed.

The final result and supporting measurements will be added after testing.

## 8. Final Conclusion

The camera calibration, dataset verification, measurement comparison, and simulated reliability checks have been performed.

The dataset verification reported 83 verified masks with no failed mask checks. All five simulated reliability tests passed.

Measurement comparison using 10 images showed a width MAE of 3.7460 mm and a height MAE of 0.9560 mm against the measured physical reference dimensions of 57 × 86 mm.

Further evaluation is required to investigate the width measurement error, complete segmentation performance reporting, perform quantitative undistortion comparison, and evaluate the second measurable objective.

The final system performance will be assessed using the completed experimental results.

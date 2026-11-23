
# Camera Calibration Report

## 1. Overview

Camera calibration was performed for the Measure It project using OpenCV and a ChArUco calibration board. The purpose was to estimate the camera intrinsic matrix and lens distortion coefficients so that captured images could be corrected before object segmentation and physical measurement.

The selected experiment uses portrait-oriented calibration images with the third radial distortion coefficient (k3) fixed at zero.

## 2. Calibration Setup

| Parameter | Value |
|---|---|
| Calibration method | ChArUco |
| OpenCV version | 5.1.0 |
| Total calibration images | 45 |
| Accepted images | 35 |
| Skipped images | 10 |
| Image resolution | 3072 × 4096 pixels |
| Board configuration | 5 × 7 squares |
| Square length | 25 mm |
| Marker length | 18 mm |
| Calibration flag | `cv2.CALIB_FIX_K3` |
| Fixed k3 value | 0.0 |

The calibration used images in portrait orientation with a consistent resolution.

## 3. Camera Matrix

The estimated camera intrinsic matrix is:

```text
[[3244.48941975,    0.0,        1556.73647137],
 [   0.0,        3253.32741041, 2033.38770182],
 [   0.0,           0.0,           1.0       ]]
```

The matrix contains the focal lengths in pixel units and the principal point coordinates.

## 4. Distortion Coefficients

The estimated distortion coefficients are:

| Coefficient | Value |
|---|---:|
| k1 | 0.34695331 |
| k2 | -1.18244198 |
| p1 | -0.00799171 |
| p2 | 0.00502751 |
| k3 | 0.00000000 |

The k3 coefficient was fixed at zero during calibration using `cv2.CALIB_FIX_K3`.

## 5. Reprojection Error

The calibration produced the following results:

| Metric | Result |
|---|---:|
| OpenCV RMS reprojection error | 2.1247 pixels |
| Mean point reprojection error | 1.4205 pixels |

The reprojection error measures the difference between observed ChArUco corner positions and their corresponding reprojected positions using the estimated camera parameters.

These results describe calibration fit and should not be interpreted as the final physical measurement accuracy of the system.

## 6. Accepted and Skipped Images

Out of 45 calibration images:

- 35 images were accepted because sufficient ChArUco corners were detected.
- 10 images were skipped because of insufficient detected corners or unsuitable image orientation for this experiment.

### Accepted Images

```text
IMG20261119110604.jpg
IMG20261119110615.jpg
IMG20261119110623.jpg
IMG20261119110629.jpg
IMG20261119110639.jpg
IMG20261119110648.jpg
IMG20261119110659.jpg
IMG20261119110709.jpg
IMG20261119110715.jpg
IMG20261119110728.jpg
IMG20261119110806.jpg
IMG20261119110814.jpg
IMG20261119110839.jpg
IMG20261119110846.jpg
IMG20261119110851.jpg
IMG20261119110901.jpg
IMG20261119110914.jpg
IMG20261119110922.jpg
IMG20261119111000.jpg
IMG20261119111004.jpg
IMG20261119111009.jpg
IMG20261119111020.jpg
IMG20261119111023.jpg
IMG20261119111026.jpg
IMG20261119111030.jpg
IMG20261119111034.jpg
IMG20261119111222.jpg
IMG20261119111231.jpg
IMG20261119111239.jpg
IMG20261119111246.jpg
IMG20261119111252.jpg
IMG20261119111255.jpg
IMG20261119111259.jpg
IMG20261119111303.jpg
IMG20261119111307.jpg
```

### Skipped Images

```text
IMG20261119110741.jpg
IMG20261119110748.jpg
IMG20261119111038.jpg
IMG20261119111045.jpg
IMG20261119111047.jpg
IMG20261119111049.jpg
IMG20261119111054.jpg
IMG20261119153516.jpg
IMG20261119153517.jpg
IMG20261119153520.jpg
```

## 7. Undistortion Validation

Three dataset images were visually inspected after undistortion using the selected calibration:

1. `IMG20261119153145.jpg`
2. `IMG20261119153154.jpg`
3. `IMG20261119153211_01.jpg`

The second image contained a relatively straight card, while the third contained a tilted card.

The visual inspection reported the following observations:

- Card edges remained straight without additional bending.
- The ArUco marker retained its square appearance.
- Background edges remained visually flat.
- No significant black borders or unwanted stretching were observed.
- The tilted card retained its original orientation without additional warping.

The undistortion process used an alpha value of 0.0 with `cv2.getOptimalNewCameraMatrix()` to prioritize a valid image area.

The reported ROI for the tested images was:

```text
(0, 0, 3071, 4095)
```

These visual checks support using this calibration as the working configuration for the next development stage. They do not independently establish absolute measurement accuracy.

## 8. Physical Measurement Reference

The physical playing card was measured using a ruler. The measurements were checked multiple times to maintain consistency.

The confirmed physical reference dimensions are:

| Reference Object | Width | Height |
|---|---:|---:|
| Playing card | 57 mm | 86 mm |
| ArUco marker | 30 mm | 30 mm |

The ArUco marker size was measured across its outer black boundary, excluding the surrounding white margin.

These physical measurements are used as reference values for evaluating the AI measurement results. The AI-generated measurements are kept separate and compared against these physical reference dimensions.

## 9. Final Calibration Decision

The portrait-only calibration with k3 fixed at zero and alpha set to 0.0 has been selected for the measurement pipeline.

The saved calibration parameters are located at:

```text
results/calibration/k3_fixed_zero/camera_calibration.npz
```

The calibration report generated by the script is located at:

```text
results/calibration/k3_fixed_zero/calibration_report.json
```

The selected parameters will be used for image undistortion before segmentation and measurement.

Final measurement accuracy will be evaluated separately by comparing AI-generated measurements against manual measurements.

## 10. Conclusion

The selected calibration produced a mean point reprojection error of 1.4205 pixels. Visual validation on three dataset images showed consistent undistortion behavior for both straight and tilted card positions.

The physical reference dimensions of the playing card and ArUco marker have also been documented for measurement evaluation.

Therefore, this calibration is accepted as the working camera calibration for continued pipeline development. Its final suitability for physical measurement will be assessed through quantitative measurement validation.

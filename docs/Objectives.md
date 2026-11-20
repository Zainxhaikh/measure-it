# Project Objectives — Measure It

## Objective 1: Measurement Accuracy

Develop a system that measures the physical width and height of a playing card in millimetres using a known-size 30 mm ArUco marker.

**Evaluation:** Compare AI measurements with at least 10 manual measurements. Calculate MAE, mean percentage error, mean error, median error and worst-case error.

## Objective 2: Segmentation Performance

Train and evaluate an image segmentation model that identifies the playing card accurately.

**Evaluation:** Report IoU, mAP@0.5, mAP@0.5:0.95, precision and recall on the appropriate dataset.

## Objective 3: Camera Calibration

Calibrate the camera using ChArUco images to reduce lens distortion.

**Evaluation:** Save the camera matrix and distortion coefficients and report reprojection error. Demonstrate the effect of undistortion.

## Objective 4: Measurement Reliability

Make the system identify unreliable measurement conditions, including missing or partially visible reference markers, poor segmentation and challenging image angles.

**Evaluation:** Test at least one adverse condition and demonstrate when the system rejects an unreliable measurement.
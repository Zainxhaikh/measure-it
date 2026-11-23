# Project Objectives — Measure It

## Objective 1: Measurement Accuracy

Develop an image-based measurement system that detects a playing card and calculates its physical width and height in millimetres using a known-size 30 mm ArUco marker as a reference.

**Evaluation:**
Compare AI-generated measurements with at least 10 manual measurements using the actual card dimensions of 57 mm × 86 mm. Calculate and report Mean Absolute Error (MAE), Mean Percentage Error, Mean Error, Median Absolute Error, and Worst-case Error for both width and height.

## Objective 2: Segmentation Performance

Train and evaluate an image segmentation model that accurately identifies and segments the playing card from input images.

**Evaluation:**
Evaluate the model on the appropriate test dataset and report IoU, mAP@0.5, mAP@0.5:0.95, precision, and recall.

## Objective 3: Camera Calibration

Calibrate the camera using ChArUco images to estimate camera parameters and reduce lens distortion before performing measurements.

**Evaluation:**
Save the camera matrix and distortion coefficients, report the calibration reprojection error, and demonstrate the visual effect of image undistortion.

## Objective 4: Measurement Reliability

Develop reliability checks that identify conditions in which the system cannot produce a trustworthy measurement, including missing or partially visible reference markers, poor segmentation confidence, and challenging camera angles.

**Evaluation:**
Test the reliability checks using adverse-condition scenarios and demonstrate that the system rejects measurements when the reference marker, segmentation, or image conditions are unreliable.
# Project Plan — Measure It: Segment It, Measure It, Prove It

## 1. Project Overview

The goal of this project is to develop a computer vision system that detects and segments a playing card in an image and measures its physical dimensions in millimetres using a known-size ArUco marker. The system will use camera calibration to reduce lens distortion and will provide measurement results with visual evidence.

## 2. Implementation Plan

| Phase | Tasks | Expected Output |
|---|---|---|
| 1. Project Setup | Organize project structure, prepare documentation and configure environment | Ready-to-use project |
| 2. Camera Calibration | Detect ChArUco corners, calculate camera matrix and distortion coefficients | Saved calibration parameters and reprojection error |
| 3. Dataset Preparation | Inspect images, create segmentation masks and fixed train/validation/test splits | Prepared dataset |
| 4. Segmentation | Train an allowed segmentation model and evaluate its performance | Trained model and evaluation metrics |
| 5. Measurement Pipeline | Undistort image, detect card and ArUco marker, calculate dimensions | End-to-end measurement system |
| 6. Validation | Compare AI measurements with manual measurements and test challenging conditions | Error analysis and validation results |
| 7. Finalization | Test CLI, improve documentation and prepare demonstration | Completed project and GitHub submission |

## 3. Time Budget

| Phase | Allocated Time |
|---|---:|
| Project setup and documentation | 45 minutes |
| Camera calibration | 75 minutes |
| Dataset preparation | 75 minutes |
| Segmentation model | 120 minutes |
| Measurement pipeline | 90 minutes |
| Validation and testing | 45 minutes |
| Final documentation and submission | 30 minutes |
| **Total** | **8 hours** |

## 4. Technical Approach

- Python and OpenCV will be used for image processing, camera calibration and ArUco marker detection.
- ChArUco calibration images will be used to estimate camera parameters.
- The playing-card dataset will be used to train and evaluate a segmentation model.
- The separate 30 mm ArUco marker (ID 10, DICT_6X6_250) will serve as the known-size reference.
- The segmentation model will be selected from architectures permitted by the assessment.
- Measurement reliability will be assessed using detection confidence, marker visibility and image quality.

## 5. Risks and Priorities

Potential challenges include limited CPU training speed, inconsistent lighting, perspective distortion, inaccurate masks and partially visible markers.

If time becomes limited, priority will be given to camera calibration, a working segmentation and measurement pipeline, honest evaluation, and required documentation. Optional enhancements and interface development will be skipped.

## 6. Git Workflow

Development will take place on the `dev-test01` branch. Meaningful commits will be made throughout implementation. The final changes will be submitted through a pull request from `dev-test01` into `main`.
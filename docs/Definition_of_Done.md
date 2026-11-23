# Definition of Done — Measure It

The project will be considered complete when the following requirements are satisfied:

## 1. Camera Calibration
- [ ] At least 20 usable ChArUco calibration images processed.
- [ ] Camera matrix and distortion coefficients calculated and saved.
- [ ] Reprojection error reported.
- [ ] Undistortion demonstrated.

## 2. Dataset
- [ ] At least 70 self-captured images of the playing card.
- [ ] Original images preserved.
- [ ] Segmentation masks prepared and checked.
- [ ] Dataset split into training, validation and testing sets (70/20/10).
- [ ] Split lists committed before model training or tuning.

## 3. Segmentation Model
- [ ] An assessment-permitted segmentation architecture implemented.
- [ ] Model trained or fine-tuned using the collected dataset.
- [ ] IoU, mAP@0.5, mAP@0.5:0.95, precision and recall reported.
- [ ] Model weights saved.

## 4. Measurement Pipeline
- [ ] New image can be processed through a CLI command.
- [ ] Image undistortion implemented.
- [ ] Playing card segmented.
- [ ] 30 mm ArUco marker detected.
- [ ] Pixel measurements converted into millimetres.
- [ ] Output overlay and confidence information generated.
- [ ] Missing object or reference marker handled appropriately.

## 5. Validation
- [ ] At least 10 manual measurements compared with AI measurements.
- [ ] Required measurement error metrics calculated.
- [ ] At least one adverse condition tested.
- [ ] Unreliable measurements rejected.

## 6. Documentation and Submission
- [ ] Plan.md completed.
- [ ] Objectives.md completed.
- [ ] Definition_of_Done.md completed.
- [ ] README.md includes installation, setup, usage and hardware details.
- [ ] GitHub repository contains meaningful commits.
- [ ] Pull request created from dev-test01 into main.
- [ ] Demonstration video prepared within the required time limit.
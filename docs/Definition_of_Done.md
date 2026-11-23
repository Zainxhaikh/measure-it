# Definition of Done — Measure It

The project will be considered complete when the following requirements are satisfied:

## 1. Camera Calibration

- [ ] At least 20 usable ChArUco calibration images processed.
- [ ] Camera matrix and distortion coefficients calculated and saved.
- [ ] Reprojection error calculated and reported.
- [ ] Image undistortion implemented and demonstrated.

## 2. Dataset Preparation

- [ ] At least 70 self-captured images of the playing card collected.
- [ ] Original images preserved.
- [ ] Segmentation masks prepared and verified.
- [ ] Dataset divided into training, validation, and testing sets using a 70/20/10 split.
- [ ] Dataset split lists committed before model training or tuning.

## 3. Segmentation Model

- [ ] An assessment-permitted segmentation architecture implemented.
- [ ] Model trained or fine-tuned using the collected dataset.
- [ ] Model evaluated on the test dataset.
- [ ] IoU, mAP@0.5, mAP@0.5:0.95, precision, and recall calculated and reported.
- [ ] Trained model weights saved.

## 4. Measurement Pipeline

- [ ] A new image can be processed using a command-line interface (CLI).
- [ ] Camera undistortion applied before measurement.
- [ ] Playing card detected and segmented.
- [ ] Known-size 30 mm ArUco marker detected.
- [ ] Pixel measurements converted into millimetres.
- [ ] Measurement results, visual overlay, and confidence information generated.
- [ ] Missing playing cards or reference markers handled appropriately.

## 5. Validation and Reliability

- [ ] At least 10 manual measurements compared with AI-generated measurements.
- [ ] MAE, Mean Percentage Error, Mean Error, Median Absolute Error, and Worst-case Error calculated and reported.
- [ ] At least one adverse condition tested.
- [ ] Reliability checks evaluated for conditions such as poor segmentation, partial marker visibility, or challenging camera angles.
- [ ] Unreliable measurements rejected with an appropriate explanation.

## 6. Documentation and Submission

- [ ] `Plan.md` completed.
- [ ] `Objectives.md` completed.
- [ ] `Definition_of_Done.md` completed.
- [ ] `README.md` includes installation, environment setup, usage instructions, and hardware details.
- [ ] GitHub repository contains meaningful and organized commits.
- [ ] Pull request created from `dev-test01` into `main`.
- [ ] Demonstration video recorded within the required time limit.
- [ ] Final project files, results, and documentation uploaded to the GitHub repository.
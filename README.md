
# Measure It: Segment It, Measure It, Prove It

A computer vision project that detects and segments a playing card and estimates its physical width and height in millimetres using a known-size ArUco marker.

## Project Overview

Measure It is a computer vision system designed to measure the physical dimensions of a playing card from an image.

The system combines camera calibration, image undistortion, deep learning-based card segmentation, ArUco marker detection, and geometric measurement.

A 30 mm ArUco marker is used as a known-size reference to convert image measurements into real-world dimensions.

## Key Features

- Camera calibration using ChArUco board images.
- Lens distortion correction.
- Playing card segmentation using U-Net with a ResNet-18 encoder.
- ArUco marker detection using OpenCV.
- Physical width and height estimation in millimetres.
- Batch measurement of multiple images.
- Segmentation performance evaluation.
- Comparison against manual measurements.
- Reliability checks for challenging detection conditions.

## Project Objectives

1. Calibrate the camera and correct lens distortion.
2. Segment playing cards from input images.
3. Detect a known-size ArUco marker.
4. Estimate card dimensions in millimetres.
5. Evaluate segmentation and measurement accuracy.
6. Validate system reliability through additional tests.

## Dataset

The dataset contains 83 self-captured images of playing cards.

| Dataset Split | Images |
|---|---:|
| Training | 58 |
| Validation | 16 |
| Testing | 9 |
| Total | 83 |

- Random seed: 42
- Each image has a corresponding segmentation mask.
- The dataset is divided into training, validation, and testing subsets.

## Camera Calibration

Camera calibration was performed using a ChArUco board.

| Parameter | Value |
|---|---|
| Board dimensions | 5 × 7 squares |
| Square length | 25 mm |
| Marker length | 18 mm |
| Calibration images captured | 45 |
| Accepted images | 35 |
| Skipped images | 10 |
| Image resolution | 3072 × 4096 |
| Calibration approach | Portrait-only |
| Distortion coefficient k3 | Fixed at 0 |
| Alpha | 0.0 |
| RMS reprojection error | 2.1247 pixels |
| Mean reprojection error | 1.4205 pixels |

The resulting calibration parameters are saved in:

`results/calibration/k3_fixed_zero/`

## Card Segmentation

The segmentation model uses a U-Net architecture with a ResNet-18 encoder.

### Model Architecture

- Architecture: U-Net
- Encoder: ResNet-18
- Framework: PyTorch
- Task: Binary image segmentation
- Output: Playing card segmentation mask

### Segmentation Results

Evaluation was performed on 9 test images.

| Metric | Result |
|---|---:|
| IoU | 0.9811 |
| Precision | 0.9935 |
| Recall | 0.9874 |
| Dice Score | 0.9905 |
| mAP@0.5 | 1.0000 |
| mAP@0.5:0.95 | 1.0000 |

All 9 test images produced detected masks with no missed predictions.

Note: The mAP values are based on the project's single-object mask AP evaluation method.

### Trained Model

The best trained segmentation model is saved at:

`weights/best_card_segmentation.pth`

## Measurement Methodology

The system uses a known-size ArUco marker as a physical reference.

- Marker ID: 10
- Dictionary: DICT_6X6_250
- Physical marker size: 30 mm

The detected marker provides the reference scale for converting pixel measurements into millimetres.

The segmentation mask is used to estimate the playing card boundary and its dimensions.

## Measurement Evaluation

The system was evaluated against manual reference measurements using 10 images.

Reference card dimensions:

- Width: 57 mm
- Height: 86 mm

### Measurement Results

| Metric | Width | Height |
|---|---:|---:|
| MAE | 3.746 mm | 0.956 mm |
| Mean Error | -3.746 mm | -0.956 mm |
| Median Absolute Error | 3.740 mm | 0.895 mm |
| Worst-case Error | 4.160 mm | 1.310 mm |
| Mean Percentage Error | 6.5719% | 1.1116% |

### Measurement Analysis

The height measurement shows better accuracy than the width measurement.

The negative mean error indicates that the system underestimates both dimensions.

Width estimation requires further improvement and investigation.

## Reliability Testing

Additional reliability checks were implemented to examine challenging conditions, including:

- Low-confidence detection.
- Partial marker visibility.
- High camera tilt.
- Small card appearance.

The reliability helper passed 5 out of 5 simulated logic tests.

Note: These tests validate the implemented checking logic and should not be interpreted as five separate physical camera experiments.

## Project Structure

```text
measure_it/
│
├── calibration/
│   └── images/
│
├── dataset/
│   ├── images/
│   └── masks/
│
├── docs/
│   ├── Calibration_Report.md
│   ├── Definition_of_Done.md
│   ├── Objectives.md
│   ├── Plan.md
│   └── Testing_Report.md
│
├── results/
│   ├── calibration/
│   ├── measurement/
│   └── segmentation/
│
├── src/
│   ├── calibration/
│   ├── segmentation/
│   ├── measurement/
│   └── prepare_dataset_split.py
│
├── weights/
│   └── best_card_segmentation.pth
│
├── environment.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- OpenCV
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Albumentations
- Matplotlib
- Pillow
- Git and Git LFS

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Zainxhaikh/measure-it.git
cd measure-it
```

### 2. Create a Conda environment

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

```bash
conda activate pytorch_env
```

Alternatively, install dependencies using:

```bash
pip install -r requirements.txt
```

## Project Outputs

The project stores generated outputs in the following directories:

- `results/calibration/` — calibration reports and undistortion results.
- `results/segmentation/` — segmentation metrics, predictions, and training history.
- `results/measurement/` — annotated measurement images and manual comparison reports.

## Current Limitations

- Width measurements show higher error than height measurements.
- Measurement accuracy can be affected by marker detection and camera perspective.
- Evaluation has been performed on a limited dataset.
- Further testing with different lighting conditions, viewing angles, and additional card samples is needed.

## Future Improvements

- Improve width measurement accuracy.
- Expand the dataset with more images and variations.
- Test the system under different lighting and camera angles.
- Improve robustness against partially visible ArUco markers.
- Perform additional real-world validation.

## Author

**Zain Ul Abidin**

Computer Science Student  
Virtual University of Pakistan

GitHub: https://github.com/Zainxhaikh

---

## Project Status

Core camera calibration, segmentation, measurement, and evaluation components have been implemented and tested.

The project includes documented experimental results and identified areas for improvement.

# Measure It: Segment It, Measure It, Prove It

## Project Overview

Measure It is a computer vision project that detects, segments, and measures a playing card from an image. It combines camera calibration, deep-learning-based image segmentation, and a known-size ArUco reference marker to estimate the physical dimensions of the card in millimetres.

The system performs the following tasks:

- Corrects camera lens distortion.
- Segments the playing card from the background.
- Detects a reference marker with a known physical size.
- Estimates the card's width and height in millimetres.
- Checks measurement reliability.
- Generates visual measurement overlays and confidence information.

## Technologies Used

- Python 3.12
- OpenCV
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## Project Structure

```text
measure-it/
├── calibration/
│   └── images/
├── dataset/
│   ├── images/
│   ├── masks/
│   └── splits/
├── src/
│   ├── calibration/
│   ├── segmentation/
│   └── measurement/
├── weights/
├── results/
├── docs/
├── .gitattributes
├── .gitignore
├── README.md
├── requirements.txt
└── environment.yml
```

## Dataset

The project uses 83 self-captured images of a 10 of Diamonds playing card.

### Dataset Split

| Dataset | Images |
|---|---:|
| Training | 58 |
| Validation | 16 |
| Testing | 9 |
| **Total** | **83** |

All 83 images have corresponding manually annotated segmentation masks.

The dataset is divided into training, validation, and testing sets using an approximate 70/20/10 split.

The split lists are stored in:

```text
dataset/splits/
```

## Physical Measurement Reference

The physical reference dimensions used for measurement validation are:

| Object | Physical Dimension |
|---|---:|
| Playing card width | 57 mm |
| Playing card height | 86 mm |
| ArUco marker size | 30 × 30 mm |

The playing card dimensions were measured manually using a ruler and checked multiple times.

The ArUco marker's physical size is measured across its outer black square boundary, excluding the white margin.

The marker uses:

- Dictionary: `DICT_6X6_250`
- Marker ID: `10`
- Physical size: `30 mm`

## Camera Calibration

Camera calibration uses ChArUco board images and OpenCV to estimate camera parameters and reduce lens distortion.

### Calibration Configuration

| Parameter | Value |
|---|---|
| Board | 5 × 7 squares |
| Square size | 25 mm |
| Marker size | 18 mm |
| Calibration image resolution | 3072 × 4096 |
| Total calibration images | 45 |
| Accepted images | 35 |
| Skipped images | 10 |
| RMS reprojection error | 2.1247 pixels |
| Mean point reprojection error | 1.4205 pixels |
| Calibration configuration | k3 fixed to zero |
| Undistortion alpha | 0.0 |

### Calibration Parameters

The saved camera matrix and distortion coefficients are available at:

```text
results/calibration/k3_fixed_zero/camera_calibration.npz
```

See `docs/Calibration_Report.md` for the complete calibration details.

## Segmentation Model

A U-Net-style image segmentation architecture with a ResNet-18 encoder is implemented using PyTorch.

The model predicts a binary segmentation mask identifying the playing card in an input image.

### Model Weights Verification

Model checkpoint:

```text
weights/best_card_segmentation.pth
```

SHA-256 checksum:

```text
0a58d4004894c04b7f2ec29e76bdc27d66f1690970578eef8dd54c22526e4a1f
```

### Held-out Test Results

| Metric | Result |
|---|---:|
| IoU | 0.9811 |
| Precision | 0.9935 |
| Recall | 0.9874 |
| Dice | 0.9905 |
| mAP@0.5 | Not reported |
| mAP@0.5:0.95 | Not reported |

The reported values are the available segmentation evaluation results. mAP metrics have not been provided and should not be interpreted as measured results.

## Measurement Pipeline

The measurement pipeline combines image processing, segmentation, ArUco detection, and camera calibration.

### Processing Steps

1. Load the input image.
2. Apply camera undistortion.
3. Detect and segment the playing card.
4. Detect the known-size 30 mm ArUco marker.
5. Estimate the pixel-to-millimetre scale.
6. Calculate the playing card's width and height in millimetres.
7. Evaluate measurement reliability.
8. Generate measurement overlays and output information.

The system also includes reliability checks to reject measurements under unsuitable conditions, such as poor segmentation confidence, incomplete reference markers, excessive camera tilt, or very small detected objects.

## Measurement Accuracy Evaluation

The system was evaluated against 10 image-based
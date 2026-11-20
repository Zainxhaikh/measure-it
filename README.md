# Measure It: Segment It, Measure It, Prove It

## Project Overview

Measure It is a computer vision project that detects and measures a playing card from an image using camera calibration, deep-learning-based image segmentation, and a known-size ArUco reference marker.

The system performs the following tasks:
- Corrects camera lens distortion.
- Segments the playing card from the background.
- Detects a reference marker of known physical size.
- Estimates card dimensions in millimetres.
- Checks measurement reliability.
- Generates visual measurement overlays.

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
├── README.md
├── requirements.txt
└── environment.yml
```

## Dataset

The project uses 83 self-captured images of a 10 of Diamonds playing card.

Dataset split:
- Training: 58 images
- Validation: 16 images
- Testing: 9 images

All 83 images have corresponding manually annotated segmentation masks.

## Camera Calibration

Camera calibration uses ChArUco board images and OpenCV.

- Board: 5 × 7 squares
- Square size: 25 mm
- Marker size: 18 mm
- Calibration image resolution: 3072 × 4096
- Accepted calibration images: 35
- Skipped images: 10
- RMS reprojection error: 2.1247 pixels
- Mean point reprojection error: 1.4205 pixels

Calibration parameters:

`results/calibration/k3_fixed_zero/camera_calibration.npz`

See `docs/Calibration_Report.md` for details.

## Segmentation Model

A U-Net-style segmentation architecture with a ResNet-18 encoder is implemented using PyTorch.

The model predicts a binary mask for the playing card.

### Model Weights Verification

Model checkpoint: `weights/best_card_segmentation.pth`

SHA-256 checksum:

```text
0a58d4004894c04b7f2ec29e76bdc27d66f1690970578eef8dd54c22526e4a1f

### Held-out test results

| Metric | Result |
|---|---:|
| IoU | 0.9811 |
| Precision | 0.9935 |
| Recall | 0.9874 |
| Dice | 0.9905 |
| mAP@0.5 | 
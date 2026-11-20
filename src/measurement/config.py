
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Camera calibration
CALIBRATION_FILE = (
    PROJECT_ROOT
    / "results"
    / "calibration"
    / "k3_fixed_zero"
    / "camera_calibration.npz"
)

CALIBRATION_WIDTH = 3072
CALIBRATION_HEIGHT = 4096
UNDISTORTION_ALPHA = 0.0

# Reference marker
ARUCO_MARKER_ID = 10
ARUCO_MARKER_SIZE_MM = 30.0

# Segmentation
MODEL_WEIGHTS = PROJECT_ROOT / "weights" / "best_card_segmentation.pth"
IMAGE_SIZE = 512
SEGMENTATION_THRESHOLD = 0.5
MIN_SEGMENTATION_CONFIDENCE = 0.70

# Measurement reliability
MAX_TILT_ANGLE = 15.0
MIN_CARD_AREA_RATIO = 0.01

# Results
MEASUREMENT_RESULTS_DIR = PROJECT_ROOT / "results" / "measurement"
SEGMENTATION_RESULTS_DIR = PROJECT_ROOT / "results" / "segmentation"

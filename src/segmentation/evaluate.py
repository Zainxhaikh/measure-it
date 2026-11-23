
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import CardSegmentationDataset
from model import UNetResNet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best_card_segmentation.pth"
RESULTS_DIR = PROJECT_ROOT / "results" / "segmentation"

DEVICE = torch.device("cpu")
THRESHOLD = 0.5


def calculate_metrics(predictions, targets):
    predictions = predictions > THRESHOLD
    targets = targets > 0.5

    tp = (predictions & targets).sum().item()
    fp = (predictions & ~targets).sum().item()
    fn = (~predictions & targets).sum().item()

    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)

    return tp, fp, fn, iou, precision, recall, dice


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            "Trained model weights not found. Wait for training to finish."
        )

    test_dataset = CardSegmentationDataset("test.txt")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = UNetResNet18(pretrained=False).to(DEVICE)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    total_tp = total_fp = total_fn = 0
    total_images = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            predictions = torch.sigmoid(outputs) > THRESHOLD
            targets = masks > 0.5

            tp, fp, fn, iou, precision, recall, dice = calculate_metrics(
                predictions, targets
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_images += 1

            print(
                f"Image {total_images}: "
                f"IoU={iou:.4f}, "
                f"Precision={precision:.4f}, "
                f"Recall={recall:.4f}, "
                f"Dice={dice:.4f}"
            )

    iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)
    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    dice = (2 * total_tp) / (
        2 * total_tp + total_fp + total_fn + 1e-7
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = (
        "SEGMENTATION TEST RESULTS\n"
        "=========================\n"
        f"Test images: {total_images}\n"
        f"IoU: {iou:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"Dice: {dice:.4f}\n"
    )

    report_path = RESULTS_DIR / "test_metrics.txt"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

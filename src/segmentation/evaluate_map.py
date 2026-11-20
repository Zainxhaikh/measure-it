
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import CardSegmentationDataset
from model import UNetResNet18


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_PATH = (
    PROJECT_ROOT / "weights" / "best_card_segmentation.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results" / "segmentation"
)

DEVICE = torch.device("cpu")
THRESHOLD = 0.5


def calculate_mask_iou(predicted_mask, target_mask):
    intersection = np.logical_and(
        predicted_mask, target_mask
    ).sum()

    union = np.logical_or(
        predicted_mask, target_mask
    ).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def calculate_average_precision(detections, total_ground_truth, threshold):

    if total_ground_truth == 0:
        return 0.0

    if not detections:
        return 0.0

    # Sort predictions by confidence, highest first.
    detections = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True
    )

    true_positives = []
    false_positives = []

    for detection in detections:

        if detection["iou"] >= threshold:
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    true_positives = np.cumsum(true_positives)
    false_positives = np.cumsum(false_positives)

    # Use ALL ground-truth objects, including images
    # where the model failed to produce a prediction.
    recalls = true_positives / total_ground_truth

    precisions = true_positives / np.maximum(
        true_positives + false_positives,
        1
    )

    # Add boundary values for precision-envelope integration.
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Calculate the precision envelope.
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(
            precisions[index],
            precisions[index + 1]
        )

    changes = np.where(
        recalls[1:] != recalls[:-1]
    )[0]

    average_precision = np.sum(
        (recalls[changes + 1] - recalls[changes])
        * precisions[changes + 1]
    )

    return float(average_precision)


def main():

    print("=" * 55)
    print("SEGMENTATION mAP EVALUATION")
    print("=" * 55)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {WEIGHTS_PATH}"
        )

    dataset = CardSegmentationDataset("test.txt")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = UNetResNet18(pretrained=False).to(DEVICE)

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)
    model.eval()

    detections = []
    missed_predictions = 0

    with torch.no_grad():

        for index, (images, masks) in enumerate(loader):

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = (
                torch.sigmoid(outputs)[0, 0].cpu().numpy()
            )

            predicted_mask = probabilities > THRESHOLD
            target_mask = masks[0, 0].numpy() > 0.5

            # Every test image contains one ground-truth card.
            if not target_mask.any():
                print(
                    f"Warning: Empty ground-truth mask at index {index}"
                )
                continue

            # A missing prediction is a false negative.
            if not predicted_mask.any():

                missed_predictions += 1

                print(
                    f"Image {index + 1}: No predicted mask"
                )

                continue

            iou = calculate_mask_iou(
                predicted_mask,
                target_mask
            )

            confidence = float(
                probabilities[predicted_mask].mean()
            )

            detections.append({
                "image_index": index,
                "confidence": confidence,
                "iou": iou
            })

            print(
                f"Image {index + 1}: "
                f"IoU = {iou:.4f}, "
                f"Confidence = {confidence:.4f}"
            )

    total_ground_truth = len(dataset)

    # IoU thresholds: 0.50, 0.55, ..., 0.95
    thresholds = np.round(
        np.arange(0.50, 0.951, 0.05),
        2
    )

    ap_results = {}

    for threshold in thresholds:

        ap = calculate_average_precision(
            detections,
            total_ground_truth,
            threshold
        )

        ap_results[f"AP@{threshold:.2f}"] = ap

    map_50 = ap_results["AP@0.50"]

    map_50_95 = float(
        np.mean(list(ap_results.values()))
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = [
        "SEGMENTATION mAP EVALUATION",
        "===========================",
        "Evaluation method: Single-object mask AP",
        f"Test images: {total_ground_truth}",
        f"Ground-truth masks: {total_ground_truth}",
        f"Detected masks: {len(detections)}",
        f"Missed predictions: {missed_predictions}",
        f"Prediction threshold: {THRESHOLD}",
        "",
        "AVERAGE PRECISION RESULTS",
        "-------------------------",
    ]

    for metric, value in ap_results.items():
        report.append(
            f"{metric}: {value:.4f}"
        )

    report.extend([
        "",
        f"mAP@0.5: {map_50:.4f}",
        f"mAP@0.5:0.95: {map_50_95:.4f}",
    ])

    report_path = RESULTS_DIR / "map_metrics.txt"

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n" + "\n".join(report))

    print(
        f"\nReport saved to: {report_path}"
    )


if __name__ == "__main__":
    main()

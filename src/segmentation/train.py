
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CardSegmentationDataset
from model import UNetResNet18


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results" / "segmentation"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

BATCH_SIZE = 2
EPOCHS = 15
LEARNING_RATE = 0.001


def dice_loss(predictions, targets):

    predictions = torch.sigmoid(predictions)

    smooth = 1e-6

    intersection = (predictions * targets).sum(dim=(1, 2, 3))

    union = (
        predictions.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice_score = (2 * intersection + smooth) / (union + smooth)

    return 1 - dice_score.mean()


def calculate_iou(predictions, targets):

    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()

    intersection = (predictions * targets).sum()

    union = (
        predictions.sum()
        + targets.sum()
        - intersection
    )

    return ((intersection + 1e-6) / (union + 1e-6)).item()


def run_epoch(model, loader, optimizer, criterion, training):

    model.train() if training else model.eval()

    total_loss = 0
    total_iou = 0

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = (
                criterion(outputs, masks)
                + dice_loss(outputs, masks)
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_iou += calculate_iou(outputs.detach(), masks)

    return (
        total_loss / len(loader),
        total_iou / len(loader)
    )


def main():

    torch.manual_seed(42)

    train_dataset = CardSegmentationDataset("train.txt")
    val_dataset = CardSegmentationDataset("val.txt")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = UNetResNet18(pretrained=False).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_iou = 0

    history = []

    print("=" * 55)
    print("CARD SEGMENTATION MODEL TRAINING")
    print("=" * 55)

    for epoch in range(EPOCHS):

        train_loss, train_iou = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            training=True
        )

        val_loss, val_iou = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            training=False
        )

        history.append(
            f"{epoch + 1},{train_loss:.6f},{train_iou:.6f},"
            f"{val_loss:.6f},{val_iou:.6f}"
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train IoU: {train_iou:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_iou > best_val_iou:

            best_val_iou = val_iou

            torch.save(
                model.state_dict(),
                WEIGHTS_DIR / "best_card_segmentation.pth"
            )

    history_file = RESULTS_DIR / "training_history.csv"

    history_file.write_text(
        "epoch,train_loss,train_iou,val_loss,val_iou\n"
        + "\n".join(history),
        encoding="utf-8"
    )

    print("\nTraining completed.")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Model saved in: {WEIGHTS_DIR}")
    print(f"History saved in: {history_file}")


if __name__ == "__main__":
    main()

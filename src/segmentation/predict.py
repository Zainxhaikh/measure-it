
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import CardSegmentationDataset
from model import UNetResNet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best_card_segmentation.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "segmentation" / "predictions"

DEVICE = torch.device("cpu")
THRESHOLD = 0.5


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError("Model weights not found. Wait for training.")

    dataset = CardSegmentationDataset("test.txt")

    model = UNetResNet18(pretrained=False).to(DEVICE)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index in range(len(dataset)):
        image_tensor, mask_tensor = dataset[index]

        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0).to(DEVICE))
            prediction = torch.sigmoid(output)[0, 0].cpu().numpy()

        predicted_mask = (prediction > THRESHOLD).astype(np.uint8) * 255
        actual_mask = (mask_tensor[0].numpy() > 0.5).astype(np.uint8) * 255

        image = image_tensor.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)

        overlay = image.copy()
        overlay[predicted_mask > 0] = (
            0.5 * overlay[predicted_mask > 0]
            + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")

        axes[1].imshow(actual_mask, cmap="gray")
        axes[1].set_title("Actual Mask")

        axes[2].imshow(predicted_mask, cmap="gray")
        axes[2].set_title("Predicted Mask")

        axes[3].imshow(overlay)
        axes[3].set_title("Prediction Overlay")

        for axis in axes:
            axis.axis("off")

        plt.tight_layout()

        output_path = OUTPUT_DIR / f"prediction_{index + 1:02d}.png"
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"Saved: {output_path.name}")

    print("\nPrediction visualizations completed.")


if __name__ == "__main__":
    main()

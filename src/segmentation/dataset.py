
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "dataset" / "images"
MASK_DIR = PROJECT_ROOT / "dataset" / "masks"
SPLIT_DIR = PROJECT_ROOT / "dataset" / "splits"


class CardSegmentationDataset(Dataset):

    def __init__(self, split_file, image_size=512):

        self.image_size = image_size

        split_path = SPLIT_DIR / split_file

        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}"
            )

        self.image_paths = [
            line.strip()
            for line in split_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not self.image_paths:
            raise ValueError("Dataset split is empty.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        relative_path = self.image_paths[index]

        image_path = PROJECT_ROOT / relative_path

        mask_path = (
            MASK_DIR / f"{Path(relative_path).stem}.png"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise FileNotFoundError(mask_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        image = image.astype(np.float32) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


from pathlib import Path
import random

# Project paths
IMAGE_DIR = Path("dataset/images")
SPLIT_DIR = Path("dataset/splits")

# Reproducible random seed
SEED = 42

# Expected dataset size
EXPECTED_IMAGES = 83

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    # Find and sort all image files
    images = sorted(
        file.name
        for file in IMAGE_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(images) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} images, but found {len(images)}."
        )

    # Shuffle images consistently
    random.seed(SEED)
    random.shuffle(images)

    # Create fixed splits
    train_images = images[:58]
    val_images = images[58:74]
    test_images = images[74:]

    # Create output directory
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # Save paths relative to the project root
    splits = {
        "train.txt": train_images,
        "val.txt": val_images,
        "test.txt": test_images,
    }

    for filename, image_list in splits.items():
        output_path = SPLIT_DIR / filename

        with output_path.open("w", encoding="utf-8") as file:
            for image_name in image_list:
                file.write(f"dataset/images/{image_name}\n")

        print(f"{filename}: {len(image_list)} images")

    print("\nDataset splitting completed successfully.")
    print(f"Random seed: {SEED}")
    print(f"Total images: {len(images)}")


if __name__ == "__main__":
    main()

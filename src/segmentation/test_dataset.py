
import torch

from dataset import CardSegmentationDataset


train_dataset = CardSegmentationDataset("train.txt")
val_dataset = CardSegmentationDataset("val.txt")
test_dataset = CardSegmentationDataset("test.txt")

print("TRAIN DATASET:", len(train_dataset))
print("VALIDATION DATASET:", len(val_dataset))
print("TEST DATASET:", len(test_dataset))

image, mask = train_dataset[0]

print("\nSample verification")
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Image dtype:", image.dtype)
print("Mask dtype:", mask.dtype)
print("Image min/max:", image.min().item(), image.max().item())
print("Mask unique values:", torch.unique(mask).tolist())

assert image.shape == (3, 512, 512)
assert mask.shape == (1, 512, 512)
assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})

print("\nDataset loader verification successful.")

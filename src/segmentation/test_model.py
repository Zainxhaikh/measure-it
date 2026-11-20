
import torch

from model import UNetResNet18


device = torch.device("cpu")

model = UNetResNet18(pretrained=False)
model.to(device)

sample_input = torch.randn(1, 3, 512, 512)

with torch.no_grad():
    output = model(sample_input)

print("Input shape:", sample_input.shape)
print("Output shape:", output.shape)
print("Model parameters:", sum(p.numel() for p in model.parameters()))
print("Device:", device)

assert output.shape == (1, 1, 512, 512)

print("\nModel verification successful.")

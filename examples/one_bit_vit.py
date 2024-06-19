import torch
from bitnet import OneBitViT

# Create an instance of the OneBitViT model
v = OneBitViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
)

# Generate a random image tensor
img = torch.randn(1, 3, 256, 256)

# Pass the image through the OneBitViT model to get predictions
preds = v(img)  # (1, 1000)

# Print the predictions
print(preds)

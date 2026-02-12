# Torch and `Conv2d`

## Overview
- What `nn.Conv2d` does (2D convolution over images or feature maps)
- Typical use cases (edge detection, feature extraction in CNNs)

## Initial Example: Read an Image and Apply `Conv2d`
```python
import torch
import torch.nn as nn
import cv2

# Read the image (use the provided image file in this repo)
img = cv2.imread("data/dog.jpg")  # BGR, shape (H, W, C)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to tensor: shape becomes (C, H, W), values in [0, 1]
x = torch.from_numpy(img).float() / 255.0
x = x.permute(2, 0, 1)  # (C, H, W)

# Add batch dimension: (1, C, H, W)
x = x.unsqueeze(0)

# Define a simple Conv2d layer
conv = nn.Conv2d(
    in_channels=3,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1,
)

# Apply convolution
y = conv(x)

print("input shape:", x.shape)   # (1, 3, H, W)
print("output shape:", y.shape)  # (1, 8, H, W)
```

## Key Concepts
- Input tensor shape: `(N, C_in, H, W)`
- Output tensor shape: `(N, C_out, H_out, W_out)`
- Filters/kernels: learnable weights of shape `(C_out, C_in, kH, kW)`

## Core Parameters
- `in_channels`
- `out_channels`
- `kernel_size`
- `stride`
- `padding`
- `dilation`
- `groups`
- `bias`

## Output Size Formula
- `H_out = floor((H + 2*padding - dilation*(kH-1) - 1)/stride + 1)`
- `W_out = floor((W + 2*padding - dilation*(kW-1) - 1)/stride + 1)`

## Minimal Example
- Define a `Conv2d` layer
- Create a dummy input tensor
- Run a forward pass and inspect shapes


## Common Pitfalls
- Mismatched `in_channels` vs input `C_in`
- Incorrect padding/stride leading to unexpected output size
- Forgetting batch dimension `N`

## Output Shape Examples (Using `stride`, `padding`, `out_channels`, `batch_size`)
Example 1
- Input: `(N=4, C_in=3, H=32, W=32)`
- Params: `stride=1`, `padding=1`, `out_channels=8`, `batch_size=4`
- Kernel: `3x3` (assume `dilation=1`)
- Output: `(N=4, C_out=8, H_out=32, W_out=32)`

Example 2
- Input: `(N=2, C_in=3, H=64, W=64)`
- Params: `stride=2`, `padding=1`, `out_channels=16`, `batch_size=2`
- Kernel: `3x3` (assume `dilation=1`)
- Output: `(N=2, C_out=16, H_out=32, W_out=32)`

Example 3
- Input: `(N=1, C_in=1, H=28, W=28)`
- Params: `stride=2`, `padding=0`, `out_channels=10`, `batch_size=1`
- Kernel: `5x5` (assume `dilation=1`)
- Output: `(N=1, C_out=10, H_out=12, W_out=12)`

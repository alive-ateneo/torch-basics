# Example 1: HOG Features to a Tensor

## Objective
Convert image features (Histogram of Oriented Gradients, HOG) into a PyTorch tensor for vector-style computation.

## Basics: Tensors as Vector Computation
A tensor is a multi-dimensional array. In vector computation terms:

- A **scalar** is a 0D tensor (single number).
- A **vector** is a 1D tensor (length N).
- A **matrix** is a 2D tensor (rows x columns).
- Higher dimensions represent batches, channels, time steps, etc.

Common tensor operations that mirror vector/matrix math:

- **Creation**: `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.arange`, `torch.rand`.
- **Shape and layout**: `.shape`, `.view(...)`, `.reshape(...)`, `.squeeze()`, `.unsqueeze(...)`.
- **Type and device**: `.float()`, `.long()`, `.to(device)`.
- **Vector math**: `+`, `-`, `*`, `/` (elementwise), `torch.matmul` (matrix multiply), `torch.dot` (1D dot product).
- **Reductions**: `torch.sum`, `torch.mean`, `torch.max` (collapse dimensions).
- **Normalization** (common for features): `x = (x - x.mean()) / (x.std() + 1e-8)`.

For feature vectors, you usually want a 1D or 2D tensor:

- 1D tensor: shape `[feature_dim]` for a single example.
- 2D tensor: shape `[batch, feature_dim]` for a batch of examples.

## Reading 3D Tensor Data
3D tensors are common in vision and can be read like a stack of matrices. A typical shape is `[depth, rows, cols]` or `[channels, height, width]` in PyTorch.

Example: a 3D tensor with shape `[2, 3, 4]`:

- `t[0]` is the first 2D slice (a 3x4 matrix).
- `t[0, 1]` is a 1D row (length 4) within that slice.
- `t[0, 1, 2]` is a single scalar value.

```python
import torch

t = torch.arange(24).reshape(2, 3, 4)

print(t.shape)        # torch.Size([2, 3, 4])
print(t[0].shape)     # torch.Size([3, 4])  -> first slice
print(t[0, 1].shape)  # torch.Size([4])     -> row in the slice
print(t[0, 1, 2])     # tensor(6)           -> single value

# Slicing ranges
print(t[:, 0, :])     # all slices, first row, all columns
```

## 3D Image Tensors (Channels, Height, Width)
Images are naturally 3D: width x height with 3 color channels (RGB). Libraries often store them as `[height, width, channels]` (HWC), while PyTorch typically uses `[channels, height, width]` (CHW).

```python
import cv2
import torch

# OpenCV loads color images as HWC in BGR order
img = cv2.imread("data/dog.jpg")
if img is None:
    raise FileNotFoundError("Could not read data/dog.jpg")

print(img.shape)  # (H, W, C)

# Convert to a PyTorch tensor and move channels first (CHW)
img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

print(img_tensor.shape)      # (C, H, W)
print(img_tensor[0].shape)   # first channel (H, W)
print(img_tensor[:, 10, 20]) # all channels at pixel (y=10, x=20)
```

## HOG Features to Tensor (OpenCV)

Below is a minimal example that:
1) loads an image,
2) computes HOG features with OpenCV,
3) converts those features into a PyTorch tensor.

```python
import cv2
import torch

# 1) Load image and preprocess
img = cv2.imread("data/sample_gradient.ppm")
if img is None:
    raise FileNotFoundError("Could not read data/sample_gradient.ppm")

# HOG is commonly applied to grayscale images
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# HOG expects a fixed window size; resize to a common one
gray = cv2.resize(gray, (64, 128))

# 2) Create HOG descriptor and compute features
hog = cv2.HOGDescriptor()
features = hog.compute(gray)  # shape: (feature_dim, 1)

# 3) Convert to a PyTorch tensor
# Flatten to a 1D vector for typical feature use
features = features.flatten()  # shape: (feature_dim,)
features_tensor = torch.tensor(features, dtype=torch.float32)

print("HOG feature shape:", features_tensor.shape)
print("First 5 values:", features_tensor[:5])

# Optional: normalize for learning algorithms
features_tensor = (features_tensor - features_tensor.mean()) / (features_tensor.std() + 1e-8)
```

## Notes
- If you have a batch of images, compute HOG for each and `torch.stack` them into a 2D tensor of shape `[batch, feature_dim]`.
- For vector computation tasks (e.g., similarity, classification), HOG gives a compact feature vector that can feed into models or distance metrics.
- Ensure consistent image size before HOG so all feature vectors share the same dimension.

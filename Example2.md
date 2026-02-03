# Example 2: Feature Vector to Single Output (Trainable Parameter)

## Objective
Multiply an image feature tensor by a trainable parameter to produce a single output tensor (a scalar).

## Concept: Trainable Parameters for a Single Output
If you have a feature vector `x` with shape `[feature_dim]`, a simple way to get a single output is a linear model:

```
y = w · x + b
```

- `w` is a trainable weight vector of shape `[feature_dim]`.
- `b` is a trainable bias scalar.
- `y` is a single output value (scalar tensor).

In PyTorch, you make trainable parameters with `torch.nn.Parameter`. You can compute the output using `torch.dot` (for 1D vectors) or `torch.matmul`.

## Example: HOG Features → Single Output
This example:
1) extracts HOG features from an image,
2) creates a trainable weight vector and bias,
3) multiplies features by the weights to get a single scalar output.

```python
import cv2
import torch

# 1) Load image and compute HOG features
img = cv2.imread("data/dog.jpg")
if img is None:
    raise FileNotFoundError("Could not read data/dog.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (64, 128))

hog = cv2.HOGDescriptor()
features = hog.compute(gray).flatten()  # shape: (feature_dim,)

x = torch.tensor(features, dtype=torch.float32)

# 2) Create trainable parameters
feature_dim = x.shape[0]
weights = torch.nn.Parameter(torch.randn(feature_dim) * 0.01)
bias = torch.nn.Parameter(torch.zeros(()))  # scalar

# 3) Single output (scalar tensor)
y = torch.dot(weights, x) + bias

print("Feature dim:", feature_dim)
print("Output y shape:", y.shape)  # torch.Size([])
print("Output y:", y.item())
```

## Notes
- The output `y` is a scalar tensor; `y.item()` converts it to a Python number.
- This is equivalent to a one-unit linear layer. You could also write it as `torch.matmul(weights, x)`.
- For batches, stack feature vectors into shape `[batch, feature_dim]` and use `x @ weights + bias`.

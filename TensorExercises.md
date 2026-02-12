# Exercises

## Tensor Basics
1) Create a 1D tensor with values 0 to 9. Reshape it into a 2x5 matrix.
2) Create a 3D tensor with shape `[2, 3, 4]`. Print the first slice and the last column of each slice.
3) Given a tensor `x` of shape `[3, 4]`, add a new batch dimension so its shape becomes `[1, 3, 4]`.
4) Normalize a random tensor so it has zero mean and unit standard deviation.

## Image Tensors (HWC vs CHW)
1) Load `data/dog.jpg` with OpenCV and print its shape.
2) Convert the image to a PyTorch tensor and reorder it to CHW format.
3) Extract the red channel (in OpenCV, remember the channel order) and print its shape.

## HOG Feature
1) Compute HOG features for `data/dog.jpg` after resizing to `(64, 128)`.
2) Confirm the feature dimension and print the first 10 values.
3) Stack HOG features for two images into a single 2D tensor with shape `[2, feature_dim]`.

## Trainable Parameters
1) Create a trainable weight vector matching the HOG feature dimension and a bias scalar.
2) Compute a single output scalar from `x` using `torch.dot` and add bias.
3) Extend the computation to a batch of features with shape `[batch, feature_dim]`.

## Mini Project
1) Create a tiny dataset from the two images in `data/` by computing their HOG features.
2) Build a single-output linear model using trainable parameters.
3) Run a few training steps with a dummy target (e.g., 0 for one image, 1 for the other).

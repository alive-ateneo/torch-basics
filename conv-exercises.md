# Convolution Exercises

## Exercise 1: Tiny Denoiser
You’re prototyping a tiny denoiser for 32x32 grayscale images. The model is:

- `Conv2d(1, 8, kernel_size=3, padding=1)`
- `ReLU`
- `MaxPool2d(kernel_size=2, stride=2)`
- `ConvTranspose2d(8, 1, kernel_size=2, stride=2)`

Question: How many trainable parameters are in this model?

## Exercise 2: Color Compressor
You’re compressing 64x64 RGB images into a low-resolution feature map, then upsampling back. The model is:

- `Conv2d(3, 16, kernel_size=5, padding=2)`
- `ReLU`
- `MaxPool2d(kernel_size=2, stride=2)`
- `ConvTranspose2d(16, 3, kernel_size=2, stride=2)`

Question: How many trainable parameters are in this model?

## Exercise 3: Two-Stage Upsampler
You’re building a tiny super-resolution block for 28x28 grayscale images:

- `Conv2d(1, 4, kernel_size=3, padding=1)`
- `ReLU`
- `MaxPool2d(kernel_size=2, stride=2)`
- `ConvTranspose2d(4, 2, kernel_size=2, stride=2)`
- `ReLU`
- `Conv2d(2, 1, kernel_size=3, padding=1)`

Question: How many trainable parameters are in this model?

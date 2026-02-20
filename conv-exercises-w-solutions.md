# Convolution Exercises

## Exercise 1: Tiny Denoiser
You’re prototyping a tiny denoiser for 32x32 grayscale images. The model is:

- `Conv2d(1, 8, kernel_size=3, padding=1)`
- `ReLU`
- `MaxPool2d(kernel_size=2, stride=2)`
- `ConvTranspose2d(8, 1, kernel_size=2, stride=2)`

Question: How many trainable parameters are in this model?

**Solution**
- `Conv2d(1, 8, 3x3)` params: `8 * (1*3*3) + 8` = `8*9 + 8` = `80`
- `ConvTranspose2d(8, 1, 2x2)` params: `1 * (8*2*2) + 1` = `1*32 + 1` = `33`
- `ReLU` and `MaxPool2d`: `0`
- Total: `80 + 33 = 113` parameters
- Sample input tensor:

```python
x = torch.randn(1, 1, 32, 32)
```

## Exercise 2: Color Compressor
You’re compressing 64x64 RGB images into a low-resolution feature map, then upsampling back. The model is:

- `Conv2d(3, 16, kernel_size=5, padding=2)`
- `ReLU`
- `MaxPool2d(kernel_size=2, stride=2)`
- `ConvTranspose2d(16, 3, kernel_size=2, stride=2)`

Question: How many trainable parameters are in this model?

**Solution**
- `Conv2d(3, 16, 5x5)` params: `16 * (3*5*5) + 16` = `16*75 + 16` = `1216`
- `ConvTranspose2d(16, 3, 2x2)` params: `3 * (16*2*2) + 3` = `3*64 + 3` = `195`
- `ReLU` and `MaxPool2d`: `0`
- Total: `1216 + 195 = 1411` parameters
- Sample input tensor:

```python
x = torch.randn(1, 3, 64, 64)
```

## Exercise 3: Two-Stage Upsampler
You’re building a tiny super-resolution block for 28x28 grayscale images:

- `Conv2d(1, 4, kernel_size=3, padding=1)`
- `ReLU`
- `MaxPool2d(kernel_size=2, stride=2)`
- `ConvTranspose2d(4, 2, kernel_size=2, stride=2)`
- `ReLU`
- `Conv2d(2, 1, kernel_size=3, padding=1)`

Question: How many trainable parameters are in this model?

**Solution**
- `Conv2d(1, 4, 3x3)` params: `4*(1*3*3) + 4` = `4*9 + 4` = `40`
- `ConvTranspose2d(4, 2, 2x2)` params: `2*(4*2*2) + 2` = `2*16 + 2` = `34`
- `Conv2d(2, 1, 3x3)` params: `1*(2*3*3) + 1` = `1*18 + 1` = `19`
- `ReLU` and `MaxPool2d`: `0`
- Total: `40 + 34 + 19 = 93` parameters
- Sample input tensor:

```python
x = torch.randn(1, 1, 28, 28)
```

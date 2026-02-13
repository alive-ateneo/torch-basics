```python
import cv2
import torch

# 1) Read image with OpenCV (BGR by default)
img_bgr = cv2.imread("path/to/image.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2) Resize and convert to float tensor in [0,1]
img_rgb = cv2.resize(img_rgb, (128, 128))
x = torch.from_numpy(img_rgb).float() / 255.0   # [H, W, C]
x = x.permute(2, 0, 1).unsqueeze(0)             # [1, C, H, W]

# 3) Conv2d
conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
y = conv(x)  # [1, 8, 128, 128]

# 4) Flatten (keep batch dim)
flat = torch.flatten(y, start_dim=1)  # [1, 8*128*128]

print(x.shape, y.shape, flat.shape)
```

import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# Load mô hình
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl14.pth', map_location=device))
model.to(device)
model.eval()

# Đọc ảnh đầu vào
image = cv2.imread('input.jpg')

# Tạo depth map
depth = model.infer_image(image)

# Chuyển đổi ảnh depth sang 8-bit (0-255) để lưu dưới dạng PNG
depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255
depth_norm = depth_norm.astype(np.uint8)

# Lưu ảnh PNG depth map
cv2.imwrite('depth_map.png', depth_norm)
print("✅ Đã tạo depth_map.png thành công!")

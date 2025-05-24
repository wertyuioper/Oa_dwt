

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读取图像并转换为灰度图
image_path = "C:/Users\RT\Desktop\DRCNN\dwt/1/flower-344109_640 (1).jpg"  # 替换为你的图像路径''
image = Image.open(image_path).convert("L")
image_np = np.array(image)

# 对图像进行 FFT 变换
f_transform = np.fft.fft2(image_np)
f_shift = np.fft.fftshift(f_transform)  # 将零频率分量移到中心
magnitude_spectrum = 20 * np.log(np.abs(f_shift))

# 显示频域图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_np, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Frequency Domain")
plt.imshow(magnitude_spectrum, cmap="gray")
plt.axis("off")
plt.show()

# 设定方向感应阈值进行去噪
rows, cols = image_np.shape
crow, ccol = rows // 2, cols // 2  # 中心点

# 创建方向感应掩模
threshold = 50  # 控制频率范围
angle_range = np.pi / 4  # 控制方向范围 (以弧度为单位)
mask = np.ones((rows, cols), np.uint8)
y, x = np.ogrid[:rows, :cols]
dx, dy = x - ccol, y - crow
angles = np.arctan2(dy, dx)  # 计算角度
frequencies = np.sqrt(dx**2 + dy**2)  # 计算频率幅度

# 应用方向感应阈值
mask[(frequencies < threshold) | (np.abs(angles) > angle_range)] = 0

# 应用掩模
f_shift_filtered = f_shift * mask

# 反傅里叶变换以恢复图像
f_ishift = np.fft.ifftshift(f_shift_filtered)
image_denoised = np.fft.ifft2(f_ishift)
image_denoised = np.abs(image_denoised)

# 显示去噪后的图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Filtered Frequency Domain")
plt.imshow(20 * np.log(np.abs(f_shift_filtered) + 1), cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Denoised Image")
plt.imshow(image_denoised, cmap="gray")
plt.axis("off")
plt.show()

# 保存去噪后的图像
output_path = "denoised_image.jpg"  # 替换为你的输出路径
Image.fromarray(image_denoised.astype(np.uint8)).save(output_path)

print(f"Denoised image saved to {output_path}")

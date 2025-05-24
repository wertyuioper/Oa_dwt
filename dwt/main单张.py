import cv2
import numpy as np
from utils import compute_psnr, compute_ssim, wavelet_decompose_reconstruct_color

# 读取图像（彩色）
input_image_path = 'C:/Users\RT\Desktop\DRCNN\dwt\sample_images/1.jpg'
original_image = cv2.imread(input_image_path)

# 检查图像是否加载成功
if original_image is None:
    raise FileNotFoundError(f"无法加载图像: {input_image_path}. 请检查文件路径和文件完整性。")

# 设置小波参数
wavelet = 'haar'  # Daubechies小波
level = 2        # 分解层数

# 小波分解与重构（针对彩色图像）
reconstructed_image = wavelet_decompose_reconstruct_color(original_image, wavelet, level)

# 计算PSNR和SSIM值
original_psnr = compute_psnr(original_image, original_image)  # 原始图像PSNR
reconstructed_psnr = compute_psnr(original_image, reconstructed_image)  # 重构后图像PSNR
reconstructed_ssim = compute_ssim(original_image, reconstructed_image)  # 重构后图像SSIM

# 输出PSNR和SSIM值
print(f"Original Image PSNR: {original_psnr:.2f} dB")
print(f"Reconstructed Image PSNR: {reconstructed_psnr:.2f} dB")
print(f"Reconstructed Image SSIM: {reconstructed_ssim:.4f}")

# 显示原始与处理后图像
cv2.imshow('Original Image', original_image)
cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import os
import cv2
import numpy as np
from utils import compute_psnr, compute_ssim, wavelet_decompose_reconstruct_color

# 输入和输出文件夹路径
input_folder = 'C:/Users/RT/Desktop/Oa_dwt/dwt/sample_images/'  # 输入图像文件夹
output_folder = 'C:/Users/RT/Desktop/Oa_dwt/dwt/denoised_images/'  # 输出去噪图像文件夹

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置小波参数
wavelet = 'db1'  # Daubechies小波
level = 2  # 分解层数

# 获取输入文件夹中的所有文件
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# 遍历每个图像文件
for image_file in image_files:
    input_image_path = os.path.join(input_folder, image_file)

    # 读取图像
    original_image = cv2.imread(input_image_path)

    # 检查图像是否加载成功
    if original_image is None:
        print(f"无法加载图像: {input_image_path}. 跳过该文件。")
        continue

    # 小波分解与重构（针对彩色图像）
    reconstructed_image = wavelet_decompose_reconstruct_color(original_image, wavelet, level)

    # 计算PSNR和SSIM值（可选，输出到控制台）
    original_psnr = compute_psnr(original_image, original_image)  # 原始图像PSNR
    reconstructed_psnr = compute_psnr(original_image, reconstructed_image)  # 重构后图像PSNR
    reconstructed_ssim = compute_ssim(original_image, reconstructed_image)  # 重构后图像SSIM

    print(f"处理图像: {image_file}")
    print(f"Original Image PSNR: {original_psnr:.2f} dB")
    print(f"Reconstructed Image PSNR: {reconstructed_psnr:.2f} dB")
    print(f"Reconstructed Image SSIM: {reconstructed_ssim:.4f}")

    # 保存去噪后的图像
    output_image_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_image_path, reconstructed_image)
    print(f"去噪后的图像已保存至: {output_image_path}")

print("所有图像处理完毕。")

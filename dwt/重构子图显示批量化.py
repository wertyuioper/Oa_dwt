import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2
from utils import compute_psnr, compute_ssim, wavelet_decompose_reconstruct_color
# 设置输入和输出文件夹路径
input_folder = 'C:/Users\RT\Desktop\Oa_dwt\dwt/1'
output_folder = 'C:/Users/RT/Desktop/Oa_dwt/dwt/22'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有图片文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    # 读取彩色图像
    image_path = os.path.join(input_folder, image_file)
    image1 = cv2.imread(image_path)

    # 转换为YUV色彩空间
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)

    # 小波基选择
    wavelet = 'coif5'

    # 处理亮度通道
    y_channel = image[:, :, 0]

    # 分解及重构
    coeffs_1 = pywt.dwt2(y_channel, wavelet)
    (approx_1, (horizontal_1, vertical_1, diagonal_1)) = coeffs_1

    coeffs_2 = pywt.dwt2(approx_1, wavelet)
    (approx_2, (horizontal_2, vertical_2, diagonal_2)) = coeffs_2

    coeffs_3 = pywt.dwt2(approx_2, wavelet)
    (approx_3, (horizontal_3, vertical_3, diagonal_3)) = coeffs_3

    # 小波重构
    reconstructed_3 = pywt.idwt2((approx_3, (horizontal_3, vertical_3, diagonal_3)), wavelet)
    reconstructed_2 = pywt.idwt2((approx_2, (horizontal_2, vertical_2, diagonal_2)), wavelet)
    reconstructed_1 = pywt.idwt2((approx_1, (horizontal_1, vertical_1, diagonal_1)), wavelet)

    # 确保重构后的图像与原图的亮度通道相同
    denoised_y_channel = cv2.resize(reconstructed_1, (y_channel.shape[1], y_channel.shape[0]))

    # 将降噪后的亮度通道替换回YUV图像中
    image[:, :, 0] = denoised_y_channel

    # 转换回BGR色彩空间
    denoised_bgr_image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)

    # 保存降噪后的彩色图像
    denoised_image_path = os.path.join(output_folder, image_file)
    cv2.imwrite(denoised_image_path, denoised_bgr_image)

    print(f'Saved denoised image: {denoised_image_path}')
    # 计算PSNR和SSIM值  image1是原图   image是去噪后的
    original_psnr = compute_psnr(image1, image1)  # 原始图像PSNR
    reconstructed_psnr = compute_psnr(image1, denoised_bgr_image)  # 重构后图像PSNR
    reconstructed_ssim = compute_ssim(image1, denoised_bgr_image)  # 重构后图像SSIM

    # 输出PSNR和SSIM值
    print(f"Original Image PSNR: {original_psnr:.2f} dB")
    print(f"Reconstructed Image PSNR: {reconstructed_psnr:.2f} dB")
    print(f"Reconstructed Image SSIM: {reconstructed_ssim:.4f}")

print('All images processed and saved!')

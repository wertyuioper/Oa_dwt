import os
import numpy as np
import cv2
import pywt
from utils import compute_psnr, compute_ssim

# 设置输入和输出文件夹路径
input_folder = 'C:/Users/RT/Desktop/Oa_dwt/dwt/zs/clean'
output_folder = 'C:/Users/RT/Desktop/Oa_dwt/dwt/zs/yuz'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有图片文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

def calculate_directional_threshold(high_freq, N, alpha=0.5, gamma=0.8):
  
    sigma = np.median(np.abs(high_freq)) / 0.6745  # 噪声标准差估计
    T = sigma * (np.log(N) ** gamma) * alpha      # 非线性动态阈值计算
    return T

def soft_threshold(W, T):
   
    return np.sign(W) * np.maximum(np.abs(W) - T, 0)

def traditional_threshold(W, T):
   
    return np.sign(W) * np.maximum(np.abs(W) - T, 0)

def add_gaussian_noise(image, mean=0, std=25):
   添加噪声的图像
  
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

for image_file in image_files:
    # 读取彩色图像
    image_path = os.path.join(input_folder, image_file)
    image_bgr = cv2.imread(image_path)

    # 向图像添加高斯噪声
    noisy_image_bgr = add_gaussian_noise(image_bgr)
    cv2.imwrite(os.path.join(output_folder, 'noisy_' + image_file), noisy_image_bgr)
    print(f'Saved noisy image: {os.path.join(output_folder, "noisy_" + image_file)}')


    image_yuv = cv2.cvtColor(noisy_image_bgr, cv2.COLOR_BGR2YUV)
    wavelet = 'db5'
    y_channel = image_yuv[:, :, 0]

    coeffs_1 = pywt.dwt2(y_channel, wavelet)
    (approx_1, (horizontal_1, vertical_1, diagonal_1)) = coeffs_1

    coeffs_2 = pywt.dwt2(approx_1, wavelet)
    (approx_2, (horizontal_2, vertical_2, diagonal_2)) = coeffs_2

    coeffs_3 = pywt.dwt2(approx_2, wavelet)
    (approx_3, (horizontal_3, vertical_3, diagonal_3)) = coeffs_3

    fixed_threshold = 20  # 固定阈值
    horizontal_3_traditional = traditional_threshold(horizontal_3, fixed_threshold)
    vertical_3_traditional = traditional_threshold(vertical_3, fixed_threshold)
    diagonal_3_traditional = traditional_threshold(diagonal_3, fixed_threshold)


    N3 = horizontal_3.size
    T3_h = calculate_directional_threshold(horizontal_3, N3)
    T3_v = calculate_directional_threshold(vertical_3, N3)
    T3_d = calculate_directional_threshold(diagonal_3, N3)

    horizontal_3_adaptive = soft_threshold(horizontal_3, T3_h)
    vertical_3_adaptive = soft_threshold(vertical_3, T3_v)
    diagonal_3_adaptive = soft_threshold(diagonal_3, T3_d)

    # 小波重构（传统方法）
    reconstructed_3_traditional = pywt.idwt2((approx_3, (horizontal_3_traditional, vertical_3_traditional, diagonal_3_traditional)), wavelet)
    reconstructed_2_traditional = pywt.idwt2((approx_2, (horizontal_2, vertical_2, diagonal_2)), wavelet)
    reconstructed_1_traditional = pywt.idwt2((approx_1, (horizontal_1, vertical_1, diagonal_1)), wavelet)

    denoised_y_channel_traditional = cv2.resize(reconstructed_1_traditional, (y_channel.shape[1], y_channel.shape[0]))

    # 小波重构（方向感知方法）
    reconstructed_3_adaptive = pywt.idwt2((approx_3, (horizontal_3_adaptive, vertical_3_adaptive, diagonal_3_adaptive)), wavelet)
    reconstructed_2_adaptive = pywt.idwt2((approx_2, (horizontal_2, vertical_2, diagonal_2)), wavelet)
    reconstructed_1_adaptive = pywt.idwt2((approx_1, (horizontal_1, vertical_1, diagonal_1)), wavelet)

    denoised_y_channel_adaptive = cv2.resize(reconstructed_1_adaptive, (y_channel.shape[1], y_channel.shape[0]))


    image_yuv_traditional = image_yuv.copy()
    image_yuv_traditional[:, :, 0] = denoised_y_channel_traditional
    denoised_bgr_image_traditional = cv2.cvtColor(image_yuv_traditional, cv2.COLOR_YUV2BGR)

    # 将降噪后的亮度通道替换回YUV图像中（方向感知方法）
    image_yuv[:, :, 0] = denoised_y_channel_adaptive
    denoised_bgr_image_adaptive = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    # 保存传统方法和方向感知方法降噪后的彩色图像
    traditional_path = os.path.join(output_folder, 'traditional_' + image_file)
    adaptive_path = os.path.join(output_folder, 'adaptive_' + image_file)
    cv2.imwrite(traditional_path, denoised_bgr_image_traditional)
    cv2.imwrite(adaptive_path, denoised_bgr_image_adaptive)

    print(f"Saved traditional method image: {traditional_path}")
    print(f"Saved adaptive method image: {adaptive_path}")

    # 生成对比图
    comparison_image = cv2.hconcat([noisy_image_bgr, denoised_bgr_image_traditional, denoised_bgr_image_adaptive])
    comparison_path = os.path.join(output_folder, 'comparison_' + image_file)
    cv2.imwrite(comparison_path, comparison_image)
    print(f"Saved comparison image: {comparison_path}")

print("All images processed and saved!")

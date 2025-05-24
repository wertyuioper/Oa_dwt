import os
import numpy as np
import cv2
import pywt
from utils import compute_psnr, compute_ssim

# 设置输入和输出文件夹路径
input_folder = 'C:/Users\RT\Desktop\DRCNN\dwt\zs\clean'
output_folder = 'C:/Users\RT\Desktop\DRCNN\dwt\zs\yuz'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有图片文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

def calculate_directional_threshold(high_freq, N, alpha=0.5, gamma=0.8):
    """
    根据高频分量计算改进的方向感知自适应阈值
    :param high_freq: 高频分量数组
    :param N: 系数数量
    :param alpha: 调节系数
    :param gamma: 非线性调整因子
    :return: 自适应阈值
    """
    sigma = np.median(np.abs(high_freq)) / 0.6745  # 噪声标准差估计
    T = sigma * (np.log(N) ** gamma) * alpha      # 非线性动态阈值计算
    return T

def soft_threshold(W, T):
    """
    软阈值处理
    :param W: 高频分量数组
    :param T: 阈值
    :return: 处理后的高频分量
    """
    return np.sign(W) * np.maximum(np.abs(W) - T, 0)

def traditional_threshold(high_freq, T):
    """
    使用固定阈值的传统阈值处理
    :param high_freq: 高频分量数组
    :param T: 固定阈值
    :return: 处理后的高频分量
    """
    return np.sign(high_freq) * np.maximum(np.abs(high_freq) - T, 0)

def add_gaussian_noise(image, mean=0, std=25):
    """
    向图像添加高斯噪声
    :param image: 输入图像
    :param mean: 噪声均值
    :param std: 噪声标准差
    :return: 添加噪声的图像
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def apply_post_filtering(image, method='bilateral'):
    """
    对降噪后的图像应用后处理步骤
    :param image: 输入图像
    :param method: 后处理方法，支持 'bilateral' 和 'gaussian'
    :return: 处理后的图像
    """
    if method == 'bilateral':
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    else:
        return image

for image_file in image_files:
    # 读取彩色图像
    image_path = os.path.join(input_folder, image_file)
    image_bgr = cv2.imread(image_path)

    # 向图像添加高斯噪声
    noisy_image_bgr = add_gaussian_noise(image_bgr)
    cv2.imwrite(os.path.join(output_folder, 'noisy_' + image_file), noisy_image_bgr)
    print(f'Saved noisy image: {os.path.join(output_folder, "noisy_" + image_file)}')

    # 转换为YUV色彩空间
    image_yuv = cv2.cvtColor(noisy_image_bgr, cv2.COLOR_BGR2YUV)

    # 小波基选择
    wavelet = 'db5'

    # 处理亮度通道
    y_channel = image_yuv[:, :, 0]

    # 分解到三级
    coeffs_1 = pywt.dwt2(y_channel, wavelet)
    (approx_1, (horizontal_1, vertical_1, diagonal_1)) = coeffs_1

    coeffs_2 = pywt.dwt2(approx_1, wavelet)
    (approx_2, (horizontal_2, vertical_2, diagonal_2)) = coeffs_2

    coeffs_3 = pywt.dwt2(approx_2, wavelet)
    (approx_3, (horizontal_3, vertical_3, diagonal_3)) = coeffs_3

    # 对高频分量应用传统阈值处理
    fixed_threshold = 20  # 固定阈值
    horizontal_3_traditional = traditional_threshold(horizontal_3, fixed_threshold)
    vertical_3_traditional = traditional_threshold(vertical_3, fixed_threshold)
    diagonal_3_traditional = traditional_threshold(diagonal_3, fixed_threshold)

    # 对高频分量应用方向感知阈值处理
    N3 = horizontal_3.size
    T3_h = calculate_directional_threshold(horizontal_3, N3)
    T3_v = calculate_directional_threshold(vertical_3, N3)
    T3_d = calculate_directional_threshold(diagonal_3, N3)

    horizontal_3 = soft_threshold(horizontal_3, T3_h)
    vertical_3 = soft_threshold(vertical_3, T3_v)
    diagonal_3 = soft_threshold(diagonal_3, T3_d)

    # 小波重构（传统方法）
    reconstructed_3_traditional = pywt.idwt2((approx_3, (horizontal_3_traditional, vertical_3_traditional, diagonal_3_traditional)), wavelet)
    reconstructed_2_traditional = pywt.idwt2((approx_2, (horizontal_2, vertical_2, diagonal_2)), wavelet)
    reconstructed_1_traditional = pywt.idwt2((approx_1, (horizontal_1, vertical_1, diagonal_1)), wavelet)

    denoised_y_channel_traditional = cv2.resize(reconstructed_1_traditional, (y_channel.shape[1], y_channel.shape[0]))

    # 小波重构（方向感知方法）
    reconstructed_3 = pywt.idwt2((approx_3, (horizontal_3, vertical_3, diagonal_3)), wavelet)
    reconstructed_2 = pywt.idwt2((approx_2, (horizontal_2, vertical_2, diagonal_2)), wavelet)
    reconstructed_1 = pywt.idwt2((approx_1, (horizontal_1, vertical_1, diagonal_1)), wavelet)

    denoised_y_channel = cv2.resize(reconstructed_1, (y_channel.shape[1], y_channel.shape[0]))

    # 将降噪后的亮度通道替换回YUV图像中（传统方法）
    image_yuv_traditional = image_yuv.copy()
    image_yuv_traditional[:, :, 0] = denoised_y_channel_traditional
    denoised_bgr_image_traditional = cv2.cvtColor(image_yuv_traditional, cv2.COLOR_YUV2BGR)

    # 将降噪后的亮度通道替换回YUV图像中（方向感知方法）
    image_yuv[:, :, 0] = denoised_y_channel
    denoised_bgr_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    # 对方向感知降噪结果应用后处理
    denoised_bgr_image_post = apply_post_filtering(denoised_bgr_image)
    post_image_path = os.path.join(output_folder, 'post_adaptive_' + image_file)
    cv2.imwrite(post_image_path, denoised_bgr_image_post)
    print(f"Saved post-processed adaptive denoised image: {post_image_path}")

    # 保存传统方法和方向感知方法降噪后的彩色图像
    denoised_image_path_traditional = os.path.join(output_folder, 'traditional_' + image_file)
    denoised_image_path = os.path.join(output_folder, 'adaptive_' + image_file)
    cv2.imwrite(denoised_image_path_traditional, denoised_bgr_image_traditional)
    cv2.imwrite(denoised_image_path, denoised_bgr_image)

    print(f'Saved traditional denoised image: {denoised_image_path_traditional}')
    print(f'Saved adaptive denoised image: {denoised_image_path}')

    # 计算PSNR和SSIM值
    reconstructed_psnr_traditional = compute_psnr(image_bgr, denoised_bgr_image_traditional)  # 传统方法PSNR
    reconstructed_psnr = compute_psnr(image_bgr, denoised_bgr_image)  # 方向感知方法PSNR
    reconstructed_ssim_traditional = compute_ssim(image_bgr, denoised_bgr_image_traditional)  # 传统方法SSIM
    reconstructed_ssim = compute_ssim(image_bgr, denoised_bgr_image)  # 方向感知方法SSIM


    # 输出PSNR和SSIM值
    print(f"Traditional Method PSNR: {reconstructed_psnr_traditional:.2f} dB")
    print(f"Adaptive Method PSNR: {reconstructed_psnr:.2f} dB")
    print(f"Traditional Method SSIM: {reconstructed_ssim_traditional:.4f}")
    print(f"Adaptive Method SSIM: {reconstructed_ssim:.4f}")

    # 生成对比图
    comparison_image = cv2.hconcat([noisy_image_bgr, denoised_bgr_image_traditional, denoised_bgr_image])

    # 保存对比图
    comparison_image_path = os.path.join(output_folder, 'comparison_' + image_file)
    cv2.imwrite(comparison_image_path, comparison_image)
    print(f'Saved comparison image: {comparison_image_path}')

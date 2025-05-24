import cv2
import numpy as np
import pywt
from skimage.metrics import structural_similarity as ssim


def compute_psnr(original, processed):
    """计算峰值信噪比（PSNR）"""
    original = cv2.resize(original, (processed.shape[1], processed.shape[0]))
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    return psnr


def compute_ssim(original, processed):
    """计算结构相似性（SSIM）"""
    original = cv2.resize(original, (processed.shape[1], processed.shape[0]))

    # 确定较小的图像尺寸并选择合适的 win_size
    min_dim = min(original.shape[0], original.shape[1])
    win_size = 7 if min_dim >= 7 else min_dim  # 确保win_size小于图像最小尺寸

    return ssim(original, processed, win_size=win_size, channel_axis=2)


def wavelet_decompose_reconstruct_color(image, wavelet, level):
    """对彩色图像的每个通道分别进行小波分解与重构"""
    channels = cv2.split(image)
    reconstructed_channels = []

    # 对每个通道分别处理
    for channel in channels:
        coeffs = pywt.wavedec2(channel, wavelet, level=level)

        # 对高频分量进行基于噪声估计的自适应阈值滤波
        filtered_coeffs = adaptive_thresholding(coeffs)

        # 重构通道
        reconstructed_channel = pywt.waverec2(filtered_coeffs, wavelet)

        # 限制重构后的像素值在0-255范围内
        reconstructed_channel = np.clip(reconstructed_channel, 0, 255).astype(np.uint8)

        # 确保通道重构后尺寸与原始一致
        reconstructed_channel = cv2.resize(reconstructed_channel, (channel.shape[1], channel.shape[0]))
        reconstructed_channels.append(reconstructed_channel)

    # 将三个通道合并回彩色图像
    reconstructed_image = cv2.merge(reconstructed_channels)
    return reconstructed_image


def adaptive_thresholding(coeffs):
    """对高频分量进行基于噪声估计的自适应阈值滤波"""
    cA = coeffs[0]  # 低频部分保持不变
    filtered_coeffs = [cA]

    # 对每一层的高频分量应用自适应阈值滤波
    for (cH, cV, cD) in coeffs[1:]:
        # 使用高频分量估计噪声标准差
        noise_sigma = estimate_noise(cH)

        # 根据噪声估计值计算阈值
        threshold = noise_sigma * np.sqrt(2 * np.log(cH.size))

        # 对每个高频分量应用软阈值
        cH_filtered = soft_threshold(cH, threshold)
        cV_filtered = soft_threshold(cV, threshold)
        cD_filtered = soft_threshold(cD, threshold)

        filtered_coeffs.append((cH_filtered, cV_filtered, cD_filtered))

    return filtered_coeffs


def estimate_noise(detail_coeffs):
    """基于高频分量估计噪声水平"""
    # 计算高频分量的中位绝对偏差 (MAD)
    return np.median(np.abs(detail_coeffs)) / 0.6745


def soft_threshold(data, threshold):
    """软阈值处理：数据超过阈值则减去阈值"""
    return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

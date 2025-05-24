import pywt
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


# 添加高斯噪声的函数
def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


# 小波去噪的函数
def wavelet_denoising(noisy_image, wavelet='db1', level=3):
    # 分通道处理
    denoised_channels = []
    for channel in cv2.split(noisy_image):
        # 小波分解
        coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=level)

        # 估计噪声标准差
        detail_coeffs = coeffs[-1][0]  # 使用最后一级细节系数
        noise_std = np.median(np.abs(detail_coeffs)) / 0.6745

        # 计算全局阈值
        def universal_threshold(coeff, noise_std):
            return noise_std * np.sqrt(2 * np.log(coeff.size))

        threshold = universal_threshold(coeffs[0], noise_std)

        # 阈值处理（软阈值）
        def soft_threshold(coeff, threshold):
            return np.sign(coeff) * np.maximum(np.abs(coeff) - threshold, 0)

        denoised_coeffs = [coeffs[0]]  # 保留近似系数
        denoised_coeffs.extend([tuple(soft_threshold(c, threshold) for c in detail) for detail in coeffs[1:]])

        # 小波重构
        denoised_channel = pywt.waverec2(denoised_coeffs, wavelet=wavelet)
        denoised_channel = np.clip(denoised_channel, 0, 255).astype(np.uint8)
        denoised_channels.append(denoised_channel)

    # 合并通道
    denoised_image = cv2.merge(denoised_channels)
    return denoised_image


# 主函数
def main():
    # 读取图像
    image = cv2.imread('/dwt/3/394990940_7af082cf8d_n.jpg')
    if image is None:
        raise FileNotFoundError("Input image not found. Please make sure 'input_image.jpg' exists.")

    # 添加噪声
    noisy_image = add_gaussian_noise(image)

    # 小波去噪并实验不同的小波基
    wavelets = ['db5', 'sym5', 'coif5']
    results = []

    for wavelet in wavelets:
        denoised_image = wavelet_denoising(noisy_image, wavelet=wavelet, level=3)
        psnr_value = psnr(image, denoised_image, data_range=255)
        results.append((wavelet, psnr_value, denoised_image))

    # 显示结果
    print("\nResults:")
    for wavelet, psnr_value, _ in results:
        print(f"Wavelet: {wavelet}, PSNR: {psnr_value:.2f}")

    # 可视化示例
    fig, axs = plt.subplots(1, len(results) + 2, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Noisy')
    axs[1].axis('off')

    for i, (wavelet, _, denoised_image) in enumerate(results):
        axs[i + 2].imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
        axs[i + 2].set_title(f"{wavelet} Denoised")
        axs[i + 2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

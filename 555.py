import cv2
import numpy as np
import os

# 图像去噪：非局部均值滤波
def denoise_image(image, h=6, template_window_size=7, search_window_size=21):#h进行调节  5一般够了
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)

# 对比度增强：自适应直方图均衡化
def adaptive_histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    if len(image.shape) == 3 and image.shape[2] == 3:  # 彩色图像
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:  # 灰度图像
        return clahe.apply(image)

# 边缘增强：Unsharp Masking
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

# 图像优化流程
def optimize_image(image):
    # 1. 去噪
    denoised_image = denoise_image(image)

    # 2. 对比度增强
    contrast_enhanced_image = adaptive_histogram_equalization(denoised_image)

    # 3. 边缘增强
    final_image = unsharp_mask(contrast_enhanced_image, amount=1.2)

    return final_image

# 批量处理图像
def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        image = cv2.imread(input_path)
        if image is None:
            print(f"无法加载图像: {filename}")
            continue

        # 图像优化
        optimized_image = optimize_image(image)

        # 保存优化后的图像
        cv2.imwrite(output_path, optimized_image)
        print(f"处理完成: {filename} -> {output_path}")

# 参数设置
input_folder = "C:/Users\RT\Desktop\DRCNN\dwt\zs\hua_50"  # 输入图像文件夹路径
output_folder = "C:/Users\RT\Desktop\DRCNN\dwt\zs\zqqq_30zs"  # 输出图像文件夹路径

process_images(input_folder, output_folder)


"""
去噪（非局部均值滤波）：
非局部均值滤波（cv2.fastNlMeansDenoisingColored）是一种高效的降噪方法，可以减少噪声同时保留边缘细节。

参数 h 决定降噪强度，默认设置为 10，可根据噪声强度调整。

对比度增强（CLAHE）：
适用于去噪后的图像，增强对比度并避免过度增强。

边缘增强（Unsharp Masking）：通过高斯模糊后与原图叠加，增强边缘的锐度，参数 amount 控制增强强度

"""

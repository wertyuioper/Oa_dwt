import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2
from utils import compute_psnr, compute_ssim, wavelet_decompose_reconstruct_color
# 读取彩色图像
image1 = cv2.imread('C:/Users\RT\Desktop\Oa_dwt\dwt/1/flower-344109_640 (1).jpg')
image = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)

# 小波基选择
wavelet = 'db1'

# 处理亮度通道
y_channel = image[:,:,0]

# 显示原图
plt.subplot(5, 3, 2)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# 分解及重构
coeffs_1 = pywt.dwt2(y_channel, wavelet)
(approx_1, (horizontal_1, vertical_1, diagonal_1)) = coeffs_1

coeffs_2 = pywt.dwt2(approx_1, wavelet)
(approx_2, (horizontal_2, vertical_2, diagonal_2)) = coeffs_2

coeffs_3 = pywt.dwt2(approx_2, wavelet)
(approx_3, (horizontal_3, vertical_3, diagonal_3)) = coeffs_3

# 第一次分解细节
plt.subplot(5, 3, 4)
plt.imshow(horizontal_1, cmap='gray')
plt.title('1st Level Horizontal Detail')

plt.subplot(5, 3, 5)
plt.imshow(vertical_1, cmap='gray')
plt.title('1st Level Vertical Detail')

plt.subplot(5, 3, 6)
plt.imshow(diagonal_1, cmap='gray')
plt.title('1st Level Diagonal Detail')

# 第二次分解细节
plt.subplot(5, 3, 7)
plt.imshow(horizontal_2, cmap='gray')
plt.title('2nd Level Horizontal Detail')

plt.subplot(5, 3, 8)
plt.imshow(vertical_2, cmap='gray')
plt.title('2nd Level Vertical Detail')

plt.subplot(5, 3, 9)
plt.imshow(diagonal_2, cmap='gray')
plt.title('2nd Level Diagonal Detail')

# 第三次分解细节
plt.subplot(5, 3, 10)
plt.imshow(horizontal_3, cmap='gray')
plt.title('3rd Level Horizontal Detail')

plt.subplot(5, 3, 11)
plt.imshow(vertical_3, cmap='gray')
plt.title('3rd Level Vertical Detail')

plt.subplot(5, 3, 12)
plt.imshow(diagonal_3, cmap='gray')
plt.title('3rd Level Diagonal Detail')

# 小波重构
reconstructed_3 = pywt.idwt2((approx_3, (horizontal_3, vertical_3, diagonal_3)), wavelet)
reconstructed_2 = pywt.idwt2((approx_2, (horizontal_2, vertical_2, diagonal_2)), wavelet)
reconstructed_1 = pywt.idwt2((approx_1, (horizontal_1, vertical_1, diagonal_1)), wavelet)

# 确保重构后的图像与原图的亮度通道相同
denoised_y_channel = cv2.resize(reconstructed_1, (y_channel.shape[1], y_channel.shape[0]))

# 将降噪后的亮度通道替换回YUV图像中
image[:,:,0] = denoised_y_channel

# 转换回BGR色彩空间
denoised_bgr_image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)

# 显示重构后的彩色图像
plt.subplot(5, 3, 14)
plt.imshow(cv2.cvtColor(denoised_bgr_image, cv2.COLOR_BGR2RGB))
plt.title('Denoised Color Image')

# 保存降噪后的彩色图像
cv2.imwrite('C:/Users/RT/Desktop/Oa_dwt/dwt/denoised_color_image.jpg', denoised_bgr_image)

# 调整子图间距以更好地展示图像
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# 显示所有图像
plt.show()
# 计算PSNR和SSIM值  image1是原图   image是去噪后的
original_psnr = compute_psnr(image1, image1)  # 原始图像PSNR
reconstructed_psnr = compute_psnr(image1, denoised_bgr_image)  # 重构后图像PSNR
reconstructed_ssim = compute_ssim(image1, denoised_bgr_image)  # 重构后图像SSIM

# 输出PSNR和SSIM值
print(f"Original Image PSNR: {original_psnr:.2f} dB")
print(f"Reconstructed Image PSNR: {reconstructed_psnr:.2f} dB")
print(f"Reconstructed Image SSIM: {reconstructed_ssim:.4f}")



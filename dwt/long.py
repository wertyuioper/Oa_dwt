import numpy as np
import cv2
import pywt

# 1. 读取原图（彩色图像）
original_image = cv2.imread('denoised_color_image.jpg')

# 将彩色图像转换为YUV色彩空间，以便独立处理亮度（Y）通道
yuv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2YUV)

# 对亮度通道进行小波降噪
wavelet_name = 'db1'
level = 3
y_channel = yuv_image[:,:,0]
coeffs_y = pywt.wavedec2(y_channel, wavelet=wavelet_name, level=level)

# 定义一个包含各个分解级别阈值的列表（您可以在这里调整阈值）
thresholds = [10, 40, 30] # 这里的阈值仅为示例，实际应用时根据需要调整

# 对每个尺度上的细节系数应用相应的阈值
coeffs_thresh_y = list(coeffs_y)
for i in range(level):
  coeffs_thresh_y[-(i+1)] = [pywt.threshold(arr, thresholds[i], 'soft') for arr in coeffs_thresh_y[-(i+1)]]
denoised_y_channel = pywt.waverec2(coeffs_thresh_y, wavelet=wavelet_name)

# 将降噪后的亮度通道替换回YUV图像中
denoised_yuv_image = yuv_image.copy()

# 确保降噪后的图像尺寸与原始图像相同
denoised_y_channel = denoised_y_channel[:denoised_yuv_image.shape[0], :denoised_yuv_image.shape[1]]

denoised_yuv_image[:,:,0] = denoised_y_channel

# 转换回BGR色彩空间
denoised_bgr_image = cv2.cvtColor(denoised_yuv_image, cv2.COLOR_YUV2BGR)

# 计算并保存原始图像和降噪后图像的差异
difference_image = abs(original_image.astype(np.float64) - denoised_bgr_image.astype(np.float64))
difference_image = cv2.convertScaleAbs(difference_image) # 将差异图像转换为8位整型便于显示和保存

# 5. 将降噪后的图片、原始图片以及差异图片保存到本地
cv2.imwrite('denoised_image.jpg', denoised_bgr_image)
cv2.imwrite('original_image.jpg', original_image)
cv2.imwrite('difference_image.jpg', difference_image)

# 加载已保存的图像（这里加载不是必须的，因为我们已经有内存中的变量，但此处为了与原逻辑保持一致）
original_image = cv2.imread('original_image.jpg')
denoised_image = cv2.imread('denoised_image.jpg')
difference_image = cv2.imread('difference_image.jpg')

# 确定新图像的宽度（假设我们希望并排显示）
new_width = original_image.shape[1] * 2 + difference_image.shape[1]

# 创建一个空白的新图像用于存放合并后的图片
combined_image = np.zeros((max(original_image.shape[0], denoised_image.shape[0], difference_image.shape[0]), new_width, 3), dtype=original_image.dtype)

# 将原图、降噪后的图片和差异图片复制到新图像上
combined_image[:original_image.shape[0], :original_image.shape[1]] = original_image
combined_image[:denoised_image.shape[0], original_image.shape[1]:original_image.shape[1]+denoised_image.shape[1]] = denoised_image
combined_image[:difference_image.shape[0], original_image.shape[1]+denoised_image.shape[1]:] = difference_image

# 设置窗口大小并显示图片
cv2.namedWindow("Original vs Denoised vs Difference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original vs Denoised vs Difference", 800, 600)

# 显示组合图像
cv2.imshow("Original vs Denoised vs Difference", combined_image)

# 添加标题
cv2.putText(combined_image, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(combined_image, "Denoised Image", (original_image.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(combined_image, "Difference Image", (original_image.shape[1] + denoised_image.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# 更新显示内容
cv2.imshow("Original vs Denoised vs Difference", combined_image)

# 按任意键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

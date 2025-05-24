import os
import cv2
import numpy as np

# 设置输入和输出文件夹路径
input_folder = 'C:/Users/RT/Desktop/Oa_dwt/dwt/zs/yuzzZ'
output_folder = 'C:/Users/RT/Desktop/Oa_dwt/dwt/zs/optimized_results'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有图片文件
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

def enhance_contrast(image, method='clahe'):
    """
    对比度增强
    :param image: 输入图像
    :param method: 对比度增强方法，支持 'histogram' 和 'clahe'
    :return: 增强后的图像
    """
    if method == 'histogram':
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    elif method == 'clahe':
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return image

def enhance_edges(image, alpha=1.5, beta=0):
    """
    边缘增强
    :param image: 输入图像
    :param alpha: 边缘增强的缩放因子
    :param beta: 亮度调整
    :return: 增强后的图像
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    edge_enhanced = cv2.convertScaleAbs(image + alpha * laplacian + beta)
    return edge_enhanced

# 设置功能开关
ENABLE_CONTRAST_ENHANCEMENT = False
ENABLE_EDGE_ENHANCEMENT = True

for image_file in image_files:
    # 读取图像
    image_path = os.path.join(input_folder, image_file)
    image_bgr = cv2.imread(image_path)

    processed_image = image_bgr

    # 对比度增强
    if ENABLE_CONTRAST_ENHANCEMENT:
        processed_image = enhance_contrast(processed_image, method='clahe')


    # 边缘增强
    if ENABLE_EDGE_ENHANCEMENT:
        processed_image = enhance_edges(processed_image)

    # 保存结果
    enhanced_path = os.path.join(output_folder, 'optimized_' + image_file)
    cv2.imwrite(enhanced_path, processed_image)

    print(f"Processed and saved optimized image: {enhanced_path}")


print("All images optimized and saved!")

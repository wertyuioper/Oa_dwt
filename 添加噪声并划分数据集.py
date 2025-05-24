import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# 设置路径
data_dir = 'C:/Users\RT\Desktop\DRCNN\dwt/1'  # 你的干净图像文件夹路径
output_dir = 'C:/Users/RT\Desktop\DRCNN\dwt/zs'  # 输出数据集的路径

# 创建输出文件夹
os.makedirs(os.path.join(output_dir, 'clean'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'tulips'), exist_ok=True)

# 获取所有干净图像
clean_images = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 随机选取一些干净图像作为带噪声图像
noisy_images = []
for img_name in clean_images:
    img_path = os.path.join(data_dir, img_name)
    img = cv2.imread(img_path)

    # 添加高斯噪声
    noise = np.random.normal(2, 2, img.shape).astype(np.uint8)  # 你可以调整噪声的标准差
    noisy_img = cv2.add(img, noise)

    # 保存干净图像和带噪声图像
    cv2.imwrite(os.path.join(output_dir, 'clean', img_name), img)
    noisy_name = f'{img_name}'  # 修改噪声图像的名称
    cv2.imwrite(os.path.join(output_dir, 'tulips', noisy_name), noisy_img)
    noisy_images.append(noisy_name)

print(f"干净图像数量: {len(clean_images)}")
print(f"生成的带噪声图像数量: {len(noisy_images)}")

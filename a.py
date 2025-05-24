import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image
import os

# 配置参数
IMAGE_DIR = 'C:/Users\RT\Desktop\DRCNN\dwt/1'  # 自然图像库路径
NOISE_IMAGE_SIZE = 512  # 噪声图像尺寸
WAVELET = 'sym5'  # 小波基类型


def calculate_energy(image):
    """计算单张图像各方向能量"""
    coeffs = pywt.dwt2(image, WAVELET)
    _, (LH, HL, HH) = coeffs

    # 计算各方向能量（按面积归一化）
    m, n = LH.shape
    E_LH = np.sum(LH ** 2) / (m * n)
    E_HL = np.sum(HL ** 2) / (m * n)
    E_HH = np.sum(HH ** 2) / (m * n)

    return E_LH, E_HL, E_HH


# 初始化累加器
total_E_LH, total_E_HL, total_E_HH = 0, 0, 0
img_count = 0

# 处理自然图像库
for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
        try:
            # 读取并转为灰度图
            img = Image.open(os.path.join(IMAGE_DIR, filename)).convert('L')
            img_array = np.array(img)

            # 累计能量
            E_LH, E_HL, E_HH = calculate_energy(img_array)
            total_E_LH += E_LH
            total_E_HL += E_HL
            total_E_HH += E_HH
            img_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# 计算平均能量
avg_E_LH = total_E_LH / img_count
avg_E_HL = total_E_HL / img_count
avg_E_HH = total_E_HH / img_count

# 归一化为比例
total = avg_E_LH + avg_E_HL + avg_E_HH
natural_ratio = np.array([avg_E_LH / total, avg_E_HL / total, avg_E_HH / total])

# 生成噪声图像
noise_img = np.random.normal(0, 50, (NOISE_IMAGE_SIZE, NOISE_IMAGE_SIZE))
E_LH_n, E_HL_n, E_HH_n = calculate_energy(noise_img)
noise_total = E_LH_n + E_HL_n + E_HH_n
noise_ratio = np.array([E_LH_n / noise_total, E_HL_n / noise_total, E_HH_n / noise_total])

# 可视化雷达图
labels = ['Horizontal', 'Vertical', 'Diagonal']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, polar=True)

# 绘制自然图像分布
values = natural_ratio.tolist() + [natural_ratio[0]]
ax.plot(angles + angles[:1], values, 'b-', label='Natural Images')
ax.fill(angles + angles[:1], values, 'b', alpha=0.1)

# 绘制噪声图像分布
values = noise_ratio.tolist() + [noise_ratio[0]]
ax.plot(angles + angles[:1], values, 'r--', label='Noise Image')
ax.fill(angles + angles[:1], values, 'r', alpha=0.1)

# 设置图形参数
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8], color="grey", size=7)
plt.ylim(0, 1)
ax.set_xticks(angles)
ax.set_xticklabels(labels)
ax.legend(loc='upper right')
plt.title("Directional Energy Distribution Comparison", pad=20)

plt.show()

# 打印数值结果
print(f"Natural Image Ratios (LH:HL:HH): {natural_ratio.round(2)}")
print(f"Noise Image Ratios (LH:HL:HH): {noise_ratio.round(2)}")
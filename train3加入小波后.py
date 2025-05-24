import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pywt
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.config.list_physical_devices('GPU'))

# 确保 TensorFlow 使用 GPU
def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"使用GPU: {gpus[0]}")
        except RuntimeError as e:
            print(e)
    else:
        print("没有检测到可用的GPU，正在使用CPU训练")


# 定义DnCNN模型
def dncnn_model(depth=17, filters=128, image_channels=3):
    input_img = Input(shape=(None, None, image_channels))
    x = Conv2D(filters, kernel_size=3, padding='same', use_bias=False)(input_img)
    x = Activation('relu')(x)

    for i in range(depth - 2):
        x = Conv2D(filters, kernel_size=3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(image_channels, kernel_size=3, padding='same', use_bias=False)(x)
    output_img = add([input_img, x])

    return Model(input_img, output_img)


# 编译模型
def compile_model(model, learning_rate=1e-4):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# 从文件夹加载图像
def load_images_from_folder(folder, target_size=(100, 100)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img / 255.0  # 归一化
            images.append(img)
    return np.array(images)


# 添加噪声
def add_noise(images, noise_factor=0.1):
    noisy_images = []
    for img in images:
        noise = np.random.normal(loc=0, scale=noise_factor ** 0.5, size=img.shape)
        noisy_img = np.clip(img + noise, 0, 1)
        noisy_images.append(noisy_img)
    return np.array(noisy_images)


# 小波分解
def wavelet_decompose_rgb(img, wavelet='haar'):
    coeffs = []
    for i in range(3):  # 对每个RGB通道分别进行小波分解
        coeffs.append(pywt.dwt2(img[:, :, i], wavelet))
    return coeffs


# 小波重构
def wavelet_reconstruct_rgb(coeffs, wavelet='haar'):
    reconstructed_img = np.zeros((coeffs[0][0].shape[0]*2, coeffs[0][0].shape[1]*2, 3))
    for i in range(3):
        reconstructed_img[:, :, i] = pywt.idwt2(coeffs[i], wavelet)
    return reconstructed_img


# 使用模型对小波子带进行去噪
def denoise_with_wavelet(img, model, wavelet='haar'):
    coeffs = wavelet_decompose_rgb(img, wavelet)
    denoised_coeffs = []

    for i in range(3):
        LL, (LH, HL, HH) = coeffs[i]

        # 扩展维度以符合模型输入要求
        LL_denoised = model.predict(np.expand_dims(np.expand_dims(LL, axis=-1), axis=0))[0]
        LH_denoised = model.predict(np.expand_dims(np.expand_dims(LH, axis=-1), axis=0))[0]
        HL_denoised = model.predict(np.expand_dims(np.expand_dims(HL, axis=-1), axis=0))[0]
        HH_denoised = model.predict(np.expand_dims(np.expand_dims(HH, axis=-1), axis=0))[0]

        denoised_coeffs.append((LL_denoised.squeeze(), (LH_denoised.squeeze(), HL_denoised.squeeze(), HH_denoised.squeeze())))

    # 重构去噪后的图像
    denoised_img = wavelet_reconstruct_rgb(denoised_coeffs, wavelet)
    return denoised_img


# 准备数据
def prepare_data():
    pristine_images = load_images_from_folder('images/')
    noisy_images = add_noise(pristine_images)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(noisy_images, pristine_images, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val


# 训练DnCNN模型
def train_dncnn_with_wavelet():
    # 确保使用GPU
    setup_gpu()

    # 准备数据
    X_train, X_val, y_train, y_val = prepare_data()

    # 构建模型
    model = dncnn_model(depth=17, filters=128, image_channels=1)  # 注意: 对应单通道的小波子带
    model = compile_model(model, learning_rate=1e-4)

    # 学习率调度器
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # 数据增强
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=10)
    datagen.fit(X_train)

    # 训练模型
    history = model.fit(datagen.flow(X_train, y_train, batch_size=16),
                        validation_data=(X_val, y_val),
                        epochs=10000,
                        callbacks=[lr_scheduler],
                        verbose=1)

    # 使用小波重构和去噪
    denoised_img = denoise_with_wavelet(X_val[0], model)

    # 保存模型
    model.save('dncnn_wavelet_color.h5')
    return history


if __name__ == "__main__":
    train_dncnn_with_wavelet()

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, add
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# 检查是否使用了 GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Available GPUs: ", tf.config.list_physical_devices('GPU'))

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
            img = cv2.resize(img, target_size)  # 调整图像大小
            img = img / 255.0  # 归一化
            images.append(img)
    return np.array(images)

# 添加噪声
def add_noise(images, noise_factor=0.1):  # 修改噪声强度
    noisy_images = []
    for img in images:
        noise = np.random.normal(loc=0, scale=noise_factor ** 0.5, size=img.shape)
        noisy_img = np.clip(img + noise, 0, 1)
        noisy_images.append(noisy_img)
    return np.array(noisy_images)

# 准备数据
def prepare_data():
    pristine_images = load_images_from_folder('images')
    noisy_images = add_noise(pristine_images)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(noisy_images, pristine_images, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# 训练DnCNN模型
def train_dncnn_model():
    # 准备数据
    X_train, X_val, y_train, y_val = prepare_data()

    # Check dimensions
    print("X_train shape after processing:", X_train.shape)
    print("y_train shape after processing:", y_train.shape)

    # 构建模型
    model = dncnn_model(depth=17, filters=128, image_channels=3)  # 滤波器数量为128
    model = compile_model(model, learning_rate=1e-4)  # 初始学习率设为1e-4

    # 学习率调度器：当验证集损失停止提升时，降低学习率
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # 数据增强
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=10)
    datagen.fit(X_train)

    # 训练模型
    history = model.fit(datagen.flow(X_train, y_train, batch_size=16),
                        validation_data=(X_val, y_val),
                        epochs=1000,
                        callbacks=[lr_scheduler],  # 添加学习率调度器
                        verbose=1)

    # 保存模型
    model.save('1000dncnn_color.h5')
    return history

if __name__ == "__main__":
    train_dncnn_model()

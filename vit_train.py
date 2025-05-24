import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# 自定义补丁划分层
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, projection_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.projection = Dense(projection_dim)  # 用于投影的全连接层

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [batch_size, -1, patches.shape[-1]])  # 展平每个补丁
        return self.projection(patches)  # 对补丁进行投影

# 自定义Vision Transformer模型
class VisionTransformer(Model):
    def __init__(self, num_patches, projection_dim, num_heads, transformer_units, num_transformer_blocks, dropout_rate, patch_size):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, projection_dim)  # 添加补丁嵌入层
        self.class_token = self.add_weight("class_token", shape=(1, 1, projection_dim))  # 类标记
        self.pos_embedding = self.add_weight("pos_embedding", shape=(1, num_patches + 1, projection_dim))  # 位置嵌入
        self.transformer_blocks = [
            self.build_transformer_block(projection_dim, num_heads, transformer_units, dropout_rate)
            for _ in range(num_transformer_blocks)
        ]
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.dense = Dense(3)  # 输出层，设置为3以匹配RGB图像

    def build_transformer_block(self, projection_dim, num_heads, transformer_units, dropout_rate):
        inputs = Input(shape=(None, projection_dim))
        # 多头自注意力层
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim
        )(inputs, inputs)
        attention_output = Dropout(dropout_rate)(attention_output)
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)  # 残差连接

        # 前馈网络
        ffn_output = Dense(transformer_units, activation='relu')(out1)
        ffn_output = Dense(projection_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        return Model(inputs=inputs, outputs=LayerNormalization(epsilon=1e-6)(out1 + ffn_output))

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x = self.patch_embedding(x)  # 将图像分割成补丁并投影
        class_token_broadcast = tf.broadcast_to(self.class_token, [batch_size, 1, self.class_token.shape[-1]])  # 广播类标记
        x = tf.concat([class_token_broadcast, x], axis=1)  # 添加类标记
        x += self.pos_embedding  # 添加位置嵌入
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = self.layer_norm(x)
        return self.dense(x)

# 加载数据集
def load_data(clean_dir, noisy_dir, img_size=(224, 224)):
    clean_images = []
    noisy_images = []

    # 加载干净图像
    for filename in os.listdir(clean_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(clean_dir, filename))
            img = cv2.resize(img, img_size)  # 调整图像大小
            clean_images.append(img)

    # 加载噪声图像
    for filename in os.listdir(noisy_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(noisy_dir, filename))
            img = cv2.resize(img, img_size)  # 调整图像大小
            noisy_images.append(img)

    clean_images = np.array(clean_images)
    noisy_images = np.array(noisy_images)

    return clean_images, noisy_images

# 训练模型
def train_vit_model():
    clean_dir = 'output_dataset/clean'
    noisy_dir = 'output_dataset/noisy'

    X_clean, X_noisy = load_data(clean_dir, noisy_dir, img_size=(224, 224))

    # 数据预处理
    X_clean = X_clean.astype('float32') / 255.0
    X_noisy = X_noisy.astype('float32') / 255.0

    # 分割训练和测试数据集
    X_train, X_val, y_train, y_val = train_test_split(X_noisy, X_clean, test_size=0.2, random_state=42)

    # 创建数据生成器
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    # 定义补丁大小和补丁数量
    patch_size = 16
    num_patches = (224 // patch_size) ** 2  # 计算补丁数量

    # 实例化Vision Transformer模型
    model = VisionTransformer(
        num_patches=num_patches,  # 补丁数量
        projection_dim=128,
        num_heads=4,
        transformer_units=128,
        num_transformer_blocks=4,
        dropout_rate=0.1,
        patch_size=patch_size  # 传入补丁大小
    )

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')

    # 训练模型
    history = model.fit(datagen.flow(X_train, y_train, batch_size=16),
                        validation_data=(X_val, y_val),
                        epochs=50)

if __name__ == "__main__":
    train_vit_model()

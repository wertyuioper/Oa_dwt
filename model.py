import tensorflow as tf
from tensorflow.keras import layers, models


def dncnn_model(depth=17, filters=64, image_channels=1, use_bnorm=True):
    model = models.Sequential()

    # 第一层：卷积+ReLU激活
    model.add(layers.Conv2D(filters, (3, 3), padding='same', input_shape=(None, None, image_channels)))
    model.add(layers.Activation('relu'))

    # 中间层：卷积+BN+ReLU激活
    for _ in range(depth - 2):
        model.add(layers.Conv2D(filters, (3, 3), padding='same'))
        if use_bnorm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

    # 最后一层：卷积
    model.add(layers.Conv2D(image_channels, (3, 3), padding='same'))

    return model


def compile_model(model):
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

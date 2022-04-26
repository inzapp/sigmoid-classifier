import os

import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, num_classes, decay):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.decay = decay

    def build(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.__conv_block(16, 3, input_layer, bn=False)
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(32, 3, x, bn=False)
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(64, 3, x, bn=False)
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(128, 3, x, bn=False)
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(256, 3, x, bn=False)
        x = self.__max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(256, 3, x, bn=False)
        y = self.__classification_layer(x)
        return tf.keras.models.Model(input_layer, y)

    def __conv_block(self, filters, kernel_size, x, bn=True):
        x = self.__conv(x, filters, kernel_size, use_bias=False if bn else True)
        if bn:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def __conv(self, x, filters, kernel_size, use_bias=True):
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer='he_normal',
            padding='same',
            use_bias=use_bias,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.decay) if self.decay > 0.0 else None)(x)

    def __classification_layer(self, x, name='output'):
        x = tf.keras.layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            kernel_initializer='glorot_normal',
            activation='sigmoid')(x)
        return tf.keras.layers.GlobalAveragePooling2D(name=name)(x)

    def __max_pool(self, x):
        return tf.keras.layers.MaxPool2D()(x)

    @staticmethod
    def __drop_filter(x, rate):
        return tf.keras.layers.SpatialDropout2D(rate)(x)

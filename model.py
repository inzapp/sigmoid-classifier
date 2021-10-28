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
        x = self.__conv_block(16, 3, input_layer, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(32, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(32, 3, x, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(64, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(64, 3, x, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(128, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(128, 3, x, bn=True)
        x = self.__avg_max_pool(x)

        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(128, 3, x, bn=False)
        x = self.__drop_filter(x, 0.0625)
        x = self.__conv_block(128, 3, x, bn=True)
        x = self.__classification_layer(x)
        return tf.keras.models.Model(input_layer, x)

    def __conv_block(self, filters, kernel_size, x, activation_first=False, bn=True):
        x = self.__conv(x, filters, kernel_size, use_bias=False if bn else True)
        if activation_first:
            x = tf.keras.layers.ReLU()(x)
            if bn:
                x = tf.keras.layers.BatchNormalization()(x)
        else:
            if bn:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        return x

    def __conv(self, x, filters, kernel_size, use_bias=True):
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.zeros(),
            padding='same',
            use_bias=use_bias,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.decay) if self.decay > 0.0 else None)(x)

    def __classification_layer(self, x, name='output'):
        x = tf.keras.layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            activation='sigmoid',
            name=name)(x)
        return tf.keras.layers.GlobalAveragePooling2D()(x)

    @staticmethod
    def __avg_max_pool(x):
        ap = tf.keras.layers.AvgPool2D()(x)
        mp = tf.keras.layers.MaxPool2D()(x)
        return tf.keras.layers.Add()([ap, mp])

    @staticmethod
    def __drop_filter(x, rate):
        return tf.keras.layers.SpatialDropout2D(rate)(x)

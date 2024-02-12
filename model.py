"""
Authors : inzapp

Github url : https://github.com/inzapp/sigmoid-classifier

Copyright 2021 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import tensorflow as tf


class Model:
    def __init__(self, input_shape, num_classes, last_conv_layer_name, cam_activation_layer_name):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.last_conv_layer_name = last_conv_layer_name
        self.cam_activation_layer_name = cam_activation_layer_name

    def build(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape, name='sc_input')
        x = self.conv2d(input_layer, 16, 3)
        if self.is_stride_over(2):
            x = self.maxpooling2d(x)

        x = self.dropout(x, 0.1)
        x = self.conv2d(x, 32, 3)
        if self.is_stride_over(4):
            x = self.maxpooling2d(x)

        x = self.dropout(x, 0.15)
        x = self.conv2d(x, 64, 3)
        if self.is_stride_over(8):
            x = self.maxpooling2d(x)

        x = self.dropout(x, 0.2)
        x = self.conv2d(x, 128, 3)
        if self.is_stride_over(16):
            x = self.maxpooling2d(x)

        x = self.dropout(x, 0.25)
        x = self.conv2d(x, 256, 3, cam_activation=True)
        if self.is_stride_over(32):
            x = self.maxpooling2d(x)

        x = self.dropout(x, 0.3)
        x = self.conv2d(x, 256, 3)
        output_layer = self.classification_layer(x)
        return tf.keras.models.Model(input_layer, output_layer)

    def is_stride_over(self, stride):
        return self.input_shape[0] >= stride and self.input_shape[1] >= stride

    def conv2d(self, x, filters, kernel_size, bn=False, cam_activation=False):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            padding='same',
            use_bias=not bn,
            kernel_size=kernel_size,
            kernel_initializer='he_normal')(x)
        if bn:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu', name=self.cam_activation_layer_name if cam_activation else None)(x)
        return x

    def classification_layer(self, x, name='sc_output'):
        x = tf.keras.layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            kernel_initializer='glorot_normal',
            activation='sigmoid',
            name=self.last_conv_layer_name)(x)
        return tf.keras.layers.GlobalAveragePooling2D(name=name)(x)

    def maxpooling2d(self, x):
        return tf.keras.layers.MaxPooling2D()(x)

    def dropout(self, x, rate):
        return tf.keras.layers.Dropout(rate)(x)


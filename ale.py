"""
Authors : inzapp

Github url : https://github.com/inzapp/absolute-logarithmic-error

Copyright 2022 inzapp Authors. All Rights Reserved.

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
import tensorflow as tf


class AbsoluteLogarithmicError(tf.keras.losses.Loss):
    """Computes the cross-entropy log scale loss between true labels and predicted labels.

    This loss function can be used regardless of classification problem or regression problem.

    See: https://github.com/inzapp/absolute-logarithmic-error

    Standalone usage:
        >>> y_true = [[0, 1], [0, 0]]
        >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
        >>> ale = AbsoluteLogarithmicError()
        >>> loss = ale(y_true, y_pred)
        >>> loss.numpy()
        array([[0.9162905, 0.9162905], [0.5108255, 0.9162905]], dtype=float32)

    Usage:
        model.compile(optimizer='sgd', loss=AbsoluteLogarithmicError())
    """
    def __init__(self, alpha=0.5, gamma=0.0, label_smoothing=0.0, reduce='none', name='AbsoluteLogarithmicError'):
        """
        Args:
            alpha: Downweight the loss where zero value positioned in y_true tensor.
            gamma: Same gamma parameter used in focal loss.
            label_smoothing: y_true tensor is clipped in range (label_smoothing, 1.0 - label_smoothing).
                for example,
                label_smoothing=0.1 : [0.0, 0.0, 1.0, 0.0] -> [0.1, 0.1, 0.9, 0.1]
                label_smoothing=0.2 : [0.0, 0.0, 1.0, 0.0] -> [0.2, 0.2, 0.8, 0.2]
            reduce:
                none: No reduce. return y_true shape loss tensor.
                mean: Reduce mean to one scalar value using all axis.
                sum: Reduce sum to one value using all axis after reduce mean using batch axis(0).
                sum_over_batch_size: Reduce sum to one value using all axis.
        """
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduce = reduce
        assert 0.0 <= self.alpha <= 0.5
        assert self.gamma == 0.0 or self.gamma >= 1.0
        assert 0.0 <= self.label_smoothing <= 0.5
        assert self.reduce in ['none', 'mean', 'sum', 'sum_over_batch_size']

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
            y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
            y_true and y_pred values must be range from 0 to 1.

        Returns:
            A `Tensor` with loss.
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        eps = tf.keras.backend.epsilon()
        y_true_clip = tf.clip_by_value(y_true, 0.0 + self.label_smoothing + eps, 1.0 - self.label_smoothing + eps)
        y_pred_clip = tf.clip_by_value(y_pred, 0.0 + eps, 1.0 - eps)
        abs_error = tf.abs(y_true_clip - y_pred_clip)
        loss = -tf.math.log((1.0 + eps) - abs_error)
        if self.gamma >= 1.0:
            alpha = tf.ones_like(y_true) * self.alpha
            alpha = tf.where(y_true == 1.0, alpha, 1.0 - alpha)
            weight = tf.pow(abs_error, self.gamma) * alpha
            loss *= weight
        if self.reduce == 'mean':
            loss = tf.reduce_mean(loss)
        elif self.reduce == 'sum':
            loss = tf.reduce_sum(tf.reduce_mean(loss, axis=0))
        elif self.reduce == 'sum_over_batch_size':
            loss = tf.reduce_sum(loss)
        return loss


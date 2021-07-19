import tensorflow as tf


class StepLRDecay(tf.keras.callbacks.Callback):
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.decay_step = epochs / 3
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 1 and (epoch - 1) % self.decay_step == 0:
            self.lr *= 0.5
            tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

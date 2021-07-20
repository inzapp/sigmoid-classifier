import os

import numpy as np
import tensorflow as tf


class CosineLRDecay(tf.keras.callbacks.Callback):

    def __init__(
            self,
            max_lr=0.1,
            min_lr=0.001,
            batch_size=32,
            cycle_length=2000,
            train_data_generator_flow=None,
            validation_data_generator_flow=None):
        self.max_val_recall = 0.0
        self.batch_sum = 0
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.cycle_length = cycle_length
        self.train_data_generator_flow = train_data_generator_flow
        self.validation_data_generator_flow = validation_data_generator_flow
        super().__init__()

    def on_train_begin(self, logs=None):
        if not (os.path.exists('checkpoints') and os.path.exists('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)

    def on_train_batch_begin(self, batch, logs=None):
        self.update(self.model)

    def update(self, model):
        self.model = model
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(((1.0 / (0.5 * self.cycle_length)) * np.pi * self.batch_sum) + np.pi))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.batch_sum += 1
        if self.batch_sum % self.cycle_length == 0:
            self.save_model()

    def save_model(self):
        if self.train_data_generator_flow is None or self.validation_data_generator_flow is None:
            self.model.save(f'checkpoints/model_{self.batch_sum}_batch.h5')
        else:
            recall = self.model.evaluate(x=self.train_data_generator_flow, batch_size=self.batch_size, return_dict=True)['recall']
            val_recall = self.model.evaluate(x=self.validation_data_generator_flow, batch_size=self.batch_size, return_dict=True)['recall']
            if val_recall > self.max_val_recall:
                self.max_val_recall = val_recall
                print(f'{self.batch_sum} batch => recall: {recall:.4f}, val_recall: {val_recall:.4f}\n')
                self.model.save(f'checkpoints/model_{self.batch_sum}_batch_recall_{recall:.4f}_val_recall_{val_recall:.4f}.h5')

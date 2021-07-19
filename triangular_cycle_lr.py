import os

import tensorflow as tf


class TriangularCycleLR(tf.keras.callbacks.Callback):

    def __init__(
            self,
            max_lr=0.01,
            min_lr=1e-5,
            batch_size=32,
            cycle_steps=2000,
            train_data_generator_flow=None,
            validation_data_generator_flow=None):
        self.batch_count = 0
        self.batch_sum = 0
        self.lr = min_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_step = cycle_steps
        self.lr_offset = (max_lr - min_lr) / float(cycle_steps / 2.0)
        self.batch_size = batch_size
        self.train_data_generator_flow = train_data_generator_flow
        self.validation_data_generator_flow = validation_data_generator_flow
        self.increasing = True
        super().__init__()

    def on_train_begin(self, logs=None):
        self.set_lr(self.min_lr)
        if not (os.path.exists('checkpoints') and os.path.exists('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)

    def on_train_batch_end(self, batch, logs=None):
        self.update(self.model)

    def update(self, model):
        self.model = model
        self.batch_sum += 1
        self.batch_count += 1
        if self.batch_count == self.cycle_step:
            self.save_model()

        if self.batch_count == int(self.cycle_step / 2 + 1):
            self.increasing = False
        elif self.batch_count == self.cycle_step + 1:
            self.increasing = True
            self.batch_count = 1
            self.max_lr *= 0.9
            self.lr_offset = (self.max_lr - self.min_lr) / float((self.cycle_step - 2) / 2.0)

        if self.increasing:
            self.increase_lr()
        else:
            self.decrease_lr()

    def increase_lr(self):
        self.lr += self.lr_offset
        self.set_lr(self.lr)

    def decrease_lr(self):
        self.lr -= self.lr_offset
        self.set_lr(self.lr)

    def set_lr(self, lr):
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def save_model(self):
        recall = self.model.evaluate(x=self.train_data_generator_flow, batch_size=self.batch_size, return_dict=True)['recall']
        val_recall = self.model.evaluate(x=self.validation_data_generator_flow, batch_size=self.batch_size, return_dict=True)['recall']
        print(f'{self.batch_sum} batch => recall: {recall:.4f}, val_recall: {val_recall:.4f}\n')
        self.model.save(f'checkpoints/model_{self.batch_sum}_batch_recall_{recall:.4f}_val_recall_{val_recall:.4f}.h5')

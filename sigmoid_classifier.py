import os
import random
from glob import glob

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from cosine_lr_decay import CosineLRDecay
from generator import SigmoidClassifierDataGenerator
from live_loss_plot import LiveLossPlot
from model import Model


class SigmoidClassifier:
    def __init__(self,
                 train_image_path,
                 input_shape,
                 max_lr,
                 min_lr,
                 burn_in,
                 cycle_length,
                 momentum,
                 batch_size,
                 max_batches,
                 pretrained_model_path='',
                 validation_image_path='',
                 validation_split=0.2):
        self.input_shape = input_shape
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.burn_in = burn_in
        self.cycle_length = cycle_length
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_batches = max_batches

        if validation_image_path != '':
            self.train_image_paths, _, self.class_names = self.__init_image_paths(train_image_path)
            self.validation_image_paths, _, self.class_names = self.__init_image_paths(validation_image_path)
        else:
            self.train_image_paths, self.validation_image_paths, self.class_names = self.__init_image_paths(train_image_path, validation_split)

        self.train_data_generator = SigmoidClassifierDataGenerator(
            image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            class_names=self.class_names)
        self.validation_data_generator = SigmoidClassifierDataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            class_names=self.class_names)

        if pretrained_model_path != '':
            self.model = tf.keras.models.load_model(pretrained_model_path, compile=False)
        else:
            self.model = Model(input_shape=self.input_shape, num_classes=len(self.class_names)).build()
        self.cosine_lr_decay = CosineLRDecay(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            cycle_length=self.cycle_length,
            train_data_generator_flow=self.train_data_generator.flow(),
            validation_data_generator_flow=self.validation_data_generator.flow())
        self.live_loss_plot = LiveLossPlot()

    @staticmethod
    def __init_image_paths(image_path, validation_split=0.0):
        dir_paths = sorted(glob(f'{image_path}/*'))
        for i in range(len(dir_paths)):
            dir_paths[i] = dir_paths[i].replace('\\', '/')
        class_name_set = set()
        train_image_paths = []
        validation_image_paths = []
        for dir_path in dir_paths:
            if not os.path.isdir(dir_path):
                continue
            dir_name = dir_path.split('/')[-1]
            if dir_name != 'unknown':
                class_name_set.add(dir_name)
            cur_class_image_paths = glob(f'{dir_path}/*.jpg') + glob(f'{dir_path}/*.png')
            for i in range(len(cur_class_image_paths)):
                cur_class_image_paths[i] = cur_class_image_paths[i].replace('\\', '/')
            if validation_split == 0.0:
                train_image_paths += cur_class_image_paths
                continue
            random.shuffle(cur_class_image_paths)
            num_cur_class_train_images = int(len(cur_class_image_paths) * (1.0 - validation_split))
            train_image_paths += cur_class_image_paths[:num_cur_class_train_images]
            validation_image_paths += cur_class_image_paths[num_cur_class_train_images:]
        class_names = sorted(list(class_name_set))
        return train_image_paths, validation_image_paths, class_names

    def evaluate(self, unknown_threshold=0.5):
        self.validation_data_generator = SigmoidClassifierDataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=1,
            class_names=self.class_names)
        num_classes = self.model.output_shape[1]
        true_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        total_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        true_unknown_count = total_unknown_count = 0
        for batch_x, batch_y in tqdm(self.validation_data_generator.flow()):
            y = self.model.predict_on_batch(x=batch_x)[0]
            # case unknown using zero label
            if np.sum(batch_y[0]) == 0.0:
                total_unknown_count += 1
                if np.max(y) < unknown_threshold:
                    true_unknown_count += 1
            # case classification
            else:
                true_class = np.argmax(batch_y[0])
                total_counts[true_class] += 1
                if np.argmax(y) == true_class:
                    true_counts[true_class] += 1

        acc_sum = 0.0
        for i in range(len(total_counts)):
            cur_class_acc = true_counts[i] / (float(total_counts[i]) + 1e-4)
            acc_sum += cur_class_acc
            print(f'[class {i:2d}] acc => {cur_class_acc:.4f}')

        valid_class_count = len(total_counts)
        if total_unknown_count != 0:
            unknown_acc = true_unknown_count / float(total_unknown_count + 1e-4)
            acc_sum += unknown_acc
            valid_class_count += 1
            print(f'[class unknown] acc => {unknown_acc:.4f}')

        acc = acc_sum / (float(valid_class_count) + 1e-4)
        print(f'sigmoid classifier accuracy with unknown threshold : {acc:.4f}')

    def fit(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.max_lr, momentum=self.momentum, nesterov=True),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        self.model.summary()
        if not (os.path.exists('checkpoints') and os.path.exists('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)

        print(f'\ntrain on {len(self.train_image_paths)} samples')
        print(f'validate on {len(self.validation_image_paths)} samples\n')
        batch_cnt = 0
        while True:
            for batch_x, batch_y in self.train_data_generator.flow():
                self.cosine_lr_decay.update(self.model)
                logs = self.model.train_on_batch(batch_x, batch_y, return_dict=True)
                self.live_loss_plot.update(logs)
                batch_cnt += 1
                if batch_cnt == self.max_batches:
                    print('train end')
                    exit(0)

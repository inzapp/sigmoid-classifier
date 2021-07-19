import os
import random
from glob import glob

import tensorflow as tf

from generator import SigmoidClassifierDataGenerator
from live_loss_plot import LiveLossPlot
from model import Model
from step_lr_decay import StepLRDecay


class SigmoidClassifier:
    def __init__(self,
                 train_image_path,
                 input_shape,
                 lr,
                 batch_size,
                 epochs,
                 pretrained_model_path='',
                 validation_image_path='',
                 validation_split=0.2):
        self.input_shape = input_shape
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

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

        self.callbacks = [LiveLossPlot(batch_range=2000)]
        self.callbacks += [StepLRDecay(lr=self.lr, epochs=self.epochs)]
        self.callbacks += [tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/model_epoch_{epoch}_recall_{recall:.4f}_val_recall_{val_recall:.4f}.h5',
            monitor='val_recall',
            mode='max',
            save_best_only=True)]

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

    def fit(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        self.model.summary()

        print(f'\ntrain on {len(self.train_image_paths)} samples')
        print(f'validate on {len(self.validation_image_paths)} samples\n')
        self.model.fit(
            x=self.train_data_generator.flow(),
            validation_data=self.validation_data_generator.flow(),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks)

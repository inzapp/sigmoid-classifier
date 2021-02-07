import os
import random
from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob

import numpy as np
import tensorflow as tf
from cv2 import cv2
from tensorflow.python.keras.utils.np_utils import to_categorical


class SigmoidClassifierDataGenerator:
    def __init__(self, train_image_path, input_shape, batch_size, validation_split=0.0):
        train_image_paths, validation_image_paths, class_names = self.__init_image_paths(train_image_path, validation_split)
        self.num_classes = len(class_names)
        self.train_generator_flow = GeneratorFlow(train_image_paths, class_names, input_shape, batch_size)
        self.validation_generator_flow = GeneratorFlow(validation_image_paths, class_names, input_shape, batch_size)

    def flow(self, subset='training'):
        if subset == 'training':
            return self.train_generator_flow
        elif subset == 'validation':
            return self.validation_generator_flow

    def get_num_classes(self):
        return self.num_classes

    @staticmethod
    def __init_image_paths(train_image_path, validation_split):
        dir_paths = sorted(glob(f'{train_image_path}/*'))
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


class GeneratorFlow(tf.keras.utils.Sequence):
    def __init__(self, image_paths, label_dict, input_shape, batch_size):
        self.image_paths = image_paths
        self.label_dict = label_dict
        self.num_classes = len(self.label_dict)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.random_indexes = np.arange(len(self.image_paths))
        self.pool = ThreadPoolExecutor(8)
        np.random.shuffle(self.random_indexes)

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        start_index = index * self.batch_size

        fs = []
        for i in range(start_index, start_index + self.batch_size):
            fs.append(self.pool.submit(self._load_img, self.image_paths[self.random_indexes[i]]))
        for f in fs:
            cur_img_path, x = f.result()
            x = cv2.resize(x, (self.input_shape[1], self.input_shape[0]))
            x = np.asarray(x).reshape(self.input_shape).astype('float32') / 255.0
            batch_x.append(x)

            dir_name = cur_img_path.split('/')[-2]
            if dir_name == 'unknown':
                y = np.zeros((self.num_classes,), dtype=np.float32)
            else:
                y = to_categorical(self.label_dict[dir_name], self.num_classes)
            batch_y.append(y)
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32')
        batch_y = np.asarray(batch_y).reshape((self.batch_size, self.num_classes)).astype('float32')
        return batch_x, batch_y

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.random_indexes)

    def _load_img(self, path):
        return path, cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.input_shape[2] == 1 else cv2.IMREAD_COLOR)

from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob

import numpy as np
import tensorflow as tf
from cv2 import cv2
from tensorflow.python.keras.utils.np_utils import to_categorical


class SigmoidClassifierDataGenerator:
    def __init__(self, train_image_path, input_shape, batch_size, validation_split=0.0):
        image_paths = self._init_image_paths(train_image_path)
        label_dict = self._init_label_dict(image_paths)
        self.num_classes = len(label_dict)
        train_image_paths, validation_image_paths = self._split_paths(image_paths, validation_split)
        self.train_generator_flow = GeneratorFlow(train_image_paths, label_dict, input_shape, batch_size, 'training')
        self.validation_generator_flow = GeneratorFlow(validation_image_paths, label_dict, input_shape, batch_size, 'validation')

    def flow(self, subset='training'):
        if subset == 'training':
            return self.train_generator_flow
        elif subset == 'validation':
            return self.validation_generator_flow

    def get_num_classes(self):
        return self.num_classes

    @staticmethod
    def _init_image_paths(train_image_path):
        image_paths = []
        image_paths += glob(f'{train_image_path}/*/*.jpg')
        image_paths += glob(f'{train_image_path}/*/*.png')
        image_paths = np.asarray(image_paths)
        for i in range(len(image_paths)):
            image_paths[i] = image_paths[i].replace('\\', '/')
        return sorted(image_paths)

    @staticmethod
    def _init_label_dict(image_paths):
        inc = 0
        label_dicts = dict()
        previous_dir_name = ''
        for path in image_paths:
            dir_name = path.split('/')[-2]
            if dir_name == 'unknown':
                continue
            if dir_name != previous_dir_name:
                previous_dir_name = dir_name
                label_dicts[dir_name] = inc
                inc += 1
        return label_dicts

    @staticmethod
    def _split_paths(image_paths, validation_split):
        assert 0.0 <= validation_split <= 1.0
        image_paths = np.asarray(image_paths)
        if validation_split == 0.0:
            return image_paths, np.asarray([])
        r = np.arange(len(image_paths))
        np.random.shuffle(r)
        image_paths = image_paths[r]
        num_train_image_paths = int(len(image_paths) * (1.0 - validation_split))
        train_image_paths = image_paths[:num_train_image_paths]
        validation_image_paths = image_paths[num_train_image_paths:]
        return train_image_paths, validation_image_paths


class GeneratorFlow(tf.keras.utils.Sequence):
    def __init__(self, image_paths, label_dict, input_shape, batch_size, subset='training'):
        self.image_paths = image_paths
        self.label_dict = label_dict
        self.num_classes = len(self.label_dict)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.subset = subset
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

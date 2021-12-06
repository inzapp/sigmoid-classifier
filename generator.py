from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from cv2 import cv2
from tensorflow.python.keras.utils.np_utils import to_categorical


class SigmoidClassifierDataGenerator:
    def __init__(self, image_paths, input_shape, batch_size, class_names):
        self.generator_flow = GeneratorFlow(image_paths, class_names, input_shape, batch_size)

    def flow(self):
        return self.generator_flow


class GeneratorFlow(tf.keras.utils.Sequence):
    def __init__(self, image_paths, class_names, input_shape, batch_size):
        self.image_paths = image_paths
        self.class_names = class_names
        self.num_classes = len(self.class_names)
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
                y = to_categorical(self.class_names.index(dir_name), self.num_classes)
            batch_y.append(y)
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32')
        batch_y = np.asarray(batch_y).reshape((self.batch_size, self.num_classes)).astype('float32')
        return batch_x, batch_y

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.random_indexes)

    def _load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.input_shape[2] == 1 else cv2.IMREAD_COLOR)
        if self.input_shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rb swap
        return path, img

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
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self, root_path, image_paths, input_shape, batch_size, class_names, augmentation=True):
        self.generator_flow = GeneratorFlow(root_path, image_paths, class_names, input_shape, batch_size, augmentation)

    def flow(self):
        return self.generator_flow


class GeneratorFlow(tf.keras.utils.Sequence):
    def __init__(self, root_path, image_paths, class_names, input_shape, batch_size, augmentation):
        self.root_path = root_path
        self.image_paths = image_paths
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0
        np.random.shuffle(self.image_paths)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.5),
            A.GaussianBlur(p=0.5, blur_limit=(7, 7))
        ])

    def __getitem__(self, index):
        fs = []
        for i in range(self.batch_size):
            fs.append(self.pool.submit(self.load_img, self.get_next_image_path()))
        batch_x = []
        batch_y = []
        for f in fs:
            img, path = f.result()
            if self.augmentation:
                img = self.transform(image=img)['image']
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            x = np.asarray(img).reshape(self.input_shape).astype('float32') / 255.0
            batch_x.append(x)

            dir_name = path.replace(self.root_path, '').split('/')[1]
            y = np.zeros((self.num_classes,), dtype=np.float32)
            if dir_name != 'unknown':
                y[self.class_names.index(dir_name)] = 1.0
            batch_y.append(y)
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32')
        batch_y = np.asarray(batch_y).reshape((self.batch_size, self.num_classes)).astype('float32')
        return batch_x, batch_y

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def get_next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

    def random_blur(self, img):
        if np.random.rand() > 0.5:
            kernel_size = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return img

    def load_img(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if self.input_shape[2] == 1 else cv2.IMREAD_COLOR)
        if self.input_shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # swap rb
        return img, path

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


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, root_path, image_paths, input_shape, batch_size, class_names, aug_brightness=0.0, aug_contrast=0.0, aug_rotate=0, aug_h_flip=False):
        assert 0.0 <= aug_brightness <= 1.0
        assert 0.0 <= aug_contrast <= 1.0
        assert type(aug_h_flip) == bool
        self.root_path = root_path
        self.image_paths = image_paths
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0
        np.random.shuffle(self.image_paths)
        aug_methods = []
        if aug_brightness > 0.0 or aug_contrast > 0.0 or aug_rotate > 0 or aug_h_flip:
            if aug_brightness > 0.0 or aug_contrast > 0.0:
                aug_methods.append(A.RandomBrightnessContrast(p=0.5, brightness_limit=aug_brightness, contrast_limit=aug_contrast))
            if aug_rotate > 0:
                aug_methods.append(A.Rotate(limit=aug_rotate, border_mode=0, value=0))
            aug_methods.append(A.GaussianBlur(p=0.5, blur_limit=(7, 7)))
        if aug_h_flip:
            aug_methods.append(A.HorizontalFlip(p=0.5))
        self.transform = A.Compose(aug_methods)
        self.augmentation = len(aug_methods) > 0

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def load(self):
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_img, self.get_next_image_path()))
        batch_x = []
        batch_y = []
        for f in fs:
            img, path = f.result()
            x = self.preprocess(img, aug=self.augmentation)
            batch_x.append(x)
            dir_name = path.replace(self.root_path, '').split('/')[1]
            y = np.zeros((self.num_classes,), dtype=np.float32)
            if dir_name != 'unknown':
                y[self.class_names.index(dir_name)] = 1.0
            batch_y.append(y)
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype(np.float32)
        batch_y = np.asarray(batch_y).reshape((self.batch_size, self.num_classes)).astype(np.float32)
        return batch_x, batch_y

    def preprocess(self, img, aug=False):
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        if aug:
            img = self.transform(image=img)['image']
        if self.input_shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # swap rb
        x = np.asarray(img).reshape(self.input_shape).astype(np.float32) / 255.0
        return x

    def get_next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

    def load_img(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if self.input_shape[-1] == 1 else cv2.IMREAD_COLOR)
        return img, path


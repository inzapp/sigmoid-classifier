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
import os
import cv2
import numpy as np
import shutil as sh
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from glob import glob
from tqdm import tqdm


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
g_save_with_score_dir = False
g_unknown_threshold = 0.5


def load_x_image_path(image_path, color_mode, input_size, input_shape):
    data = np.fromfile(image_path, dtype=np.uint8)
    x = cv2.imdecode(data, color_mode)
    x = cv2.resize(x, input_size)
    if input_shape[-1] == 3:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)  # swap rb
    x = np.asarray(x).astype('float32').reshape((1,) + input_shape) / 255.0
    return x, image_path


def auto_classification(model_path, image_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape[1:]
    input_size = (input_shape[1], input_shape[0])
    color_mode = cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR
    image_path = image_path.replace('\\', '/')
    save_path = image_path

    image_paths = glob(f'{image_path}/*.jpg')
    pool = ThreadPoolExecutor(8)
    fs = []
    for image_path in image_paths:
        fs.append(pool.submit(load_x_image_path, image_path, color_mode, input_size, input_shape))

    for f in tqdm(fs):
        x, image_path = f.result()
        y = model.predict_on_batch(x=x)[0]
        class_index = np.argmax(y)

        score = y[class_index]
        score_dir = ''
        if g_save_with_score_dir:
            score_dir = 'under_10'
            if score > 0.9:
                score_dir = 'over_90'
            elif score > 0.8:
                score_dir = 'over_80'
            elif score > 0.7:
                score_dir = 'over_70'
            elif score > 0.6:
                score_dir = 'over_60'
            elif score > 0.5:
                score_dir = 'over_50'
            elif score > 0.4:
                score_dir = 'over_40'
            elif score > 0.3:
                score_dir = 'over_30'
            elif score > 0.2:
                score_dir = 'over_20'
            elif score > 0.1:
                score_dir = 'over_10'

        if score < g_unknown_threshold:
            save_dir = f'{save_path}/unknown'
            if g_save_with_score_dir:
                save_dir += f'/{score_dir}'
            os.makedirs(save_dir, exist_ok=True)
            sh.move(image_path, save_dir)
        else:
            save_dir = f'{save_path}/{class_index}'
            if g_save_with_score_dir:
                save_dir += f'/{score_dir}'
            os.makedirs(save_dir, exist_ok=True)
            sh.move(image_path, save_dir)


def main():
    model_path = r'model.h5'
    img_path = r'/home/imagenet'
    auto_classification(model_path, img_path)


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()


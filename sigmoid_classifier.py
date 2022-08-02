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
from glob import glob
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import random
from model import Model
from generator import DataGenerator
from lr_scheduler import LRScheduler
from live_plot import LivePlot
import cv2

class SigmoidClassifier:
    def __init__(self,
                 train_image_path,
                 input_shape,
                 lr,
                 momentum,
                 batch_size,
                 iterations,
                 pretrained_model_path='',
                 validation_image_path='',
                 validation_split=0.2,
                 showcam=False,
                 activation_layer_name=None,
                 backprop_last_layer_name=None
                 ):
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.max_val_acc = 0.0
        self.showcam = showcam
        self.activation_layer_name = activation_layer_name
        self.backprop_last_layer_name = backprop_last_layer_name

        train_image_path = self.unify_path(train_image_path)
        validation_image_path = self.unify_path(validation_image_path)

        if validation_image_path != '':
            self.train_image_paths, _, self.class_names = self.init_image_paths(train_image_path)
            self.validation_image_paths, _, self.class_names = self.init_image_paths(validation_image_path)
        else:
            self.train_image_paths, self.validation_image_paths, self.class_names = self.init_image_paths(train_image_path, validation_split)

        self.train_data_generator = DataGenerator(
            root_path=train_image_path,
            image_paths=self.balance_class(train_image_path, self.train_image_paths),
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            class_names=self.class_names)
        self.validation_data_generator = DataGenerator(
            root_path=validation_image_path,
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            class_names=self.class_names)
        self.validation_data_generator_one_batch = DataGenerator(
            root_path=train_image_path if validation_image_path == '' else validation_image_path,
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=1,
            class_names=self.class_names,
            use_random_blur=False)
        self.lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations)

        if pretrained_model_path != '':
            self.model = tf.keras.models.load_model(pretrained_model_path, compile=False)
        else:
            self.model = Model(input_shape=self.input_shape, num_classes=len(self.class_names)).build()
            self.model.save('model.h5', include_optimizer=False)
        self.live_loss_plot = LivePlot(legend='loss')
        self.live_lr_plot = LivePlot(legend='learning rate', y_min=0.0, y_max=self.lr)

    def unify_path(self, path):
        if path == '':
            return path
        path = path.replace('\\', '/')
        if path.endswith('/'):
            path = path[len(self.root_path) - 1]
        return path

    def balance_class(self, root_path, paths):
        d = {}
        for path in paths:
            dir_name = path.replace(root_path, '').split('/')[1]
            try:
                d[dir_name].append(path)
            except KeyError:
                d[dir_name] = [path]
        max_length = -1
        for key in list(d.keys()):
            if len(d[key]) > max_length:
                max_length = len(d[key])
        for key in list(d.keys()):
            class_path_length = len(d[key])
            if class_path_length < max_length:
                for i in range(max_length - class_path_length):
                    random_index = np.random.randint(class_path_length)
                    d[key].append(d[key][random_index])
        new_paths = []
        for key in list(d.keys()):
            for class_image_path in d[key]:
                new_paths.append(class_image_path)
        return new_paths

    def init_image_paths(self, image_path, validation_split=0.0):
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
            if dir_name[0] == '_':
                print(f'class dir {dir_name} is ignored. dir_name[0] == "_"')
                continue
            if dir_name != 'unknown':
                class_name_set.add(dir_name)
            cur_class_image_paths = glob(f'{dir_path}/**/*.jpg', recursive=True)
            for i in range(len(cur_class_image_paths)):
                cur_class_image_paths[i] = cur_class_image_paths[i].replace('\\', '/')
            if validation_split == 0.0:
                train_image_paths += cur_class_image_paths
                continue
            np.random.shuffle(cur_class_image_paths)
            num_cur_class_train_images = int(len(cur_class_image_paths) * (1.0 - validation_split))
            train_image_paths += cur_class_image_paths[:num_cur_class_train_images]
            validation_image_paths += cur_class_image_paths[num_cur_class_train_images:]
        class_names = sorted(list(class_name_set))
        np.random.shuffle(validation_image_paths)
        return train_image_paths, validation_image_paths, class_names

    @tf.function
    def compute_gradient(self, model, optimizer, batch_x, y_true):
        with tf.GradientTape() as tape:
            y_pred = self.model(batch_x, training=True)
            loss = K.binary_crossentropy(y_true, y_pred)
            loss = tf.reduce_mean(loss, axis=0)
            mean_loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mean_loss

    def draw_cam(self, model, input_image, label):
        window_size_h = 800
        target_fmap = model.get_layer(name=self.activation_layer_name).output
        activation_h, activation_w, activation_c = target_fmap.shape[1:]
        new_model = tf.keras.Model(self.model.input, target_fmap)
        weights = model.get_layer(name=self.backprop_last_layer_name).get_weights()[0]
        weights = weights.squeeze()
        img_h, img_w, img_c = input_image.shape

        input_image = input_image[tf.newaxis, ...]
        fmap = new_model(input_image)
        display_append = None
        for idx, cls in enumerate(self.class_names):
            weights_cam = weights[:, idx]
            camsum = np.zeros((activation_h, activation_w), dtype=np.float32)
            org_image = input_image.copy()
            if img_c != 1:
                org_image = cv2.cvtColor(org_image.squeeze(0), cv2.COLOR_RGB2BGR)[tf.newaxis, ...]

            for i in range(activation_c):
                camsum += weights_cam[i] * fmap[0, :, :, i]
            camsum = camsum.numpy()
            camsum = cv2.resize(camsum, (img_w, img_h))
            camsum = ((camsum - camsum.min()) / (camsum.max() - camsum.min())) * 255
            camsum = camsum.astype(np.uint8)
            camsum = 255 - camsum
            camsum = camsum / 255.
            if img_c == 1:
                camsum = camsum[..., tf.newaxis]
            else:
                camsum = np.stack((camsum,)*3, axis=-1)

            if label == idx:
                if img_c == 1:
                    labelbox = np.ones((img_h, 20, 1), dtype=np.float32)
                else:
                    b = np.zeros((img_h, 20, 1), dtype=np.float32)
                    g = np.ones((img_h, 20, 1), dtype=np.float32)
                    r = np.zeros((img_h, 20, 1), dtype=np.float32)
                    labelbox = cv2.merge((b, g, r))
            else:
                if img_c == 1:
                    labelbox = np.zeros((img_h, 20, 1), dtype=np.float32)
                else:
                    labelbox = np.zeros((img_h, 20, 3), dtype=np.float32)
            cls_display_append = np.hstack([labelbox, org_image.squeeze(0)])
            cls_display_append = np.hstack([cls_display_append, camsum])
            alpha = 0.3
            temp_org_image = (org_image.squeeze(0) * 255).astype(np.uint8)
            temp_camsum = (camsum * 255).astype(np.uint8)
            display_blending = cv2.addWeighted(temp_org_image, alpha, temp_camsum, (1-alpha), 0)
            display_blending = cv2.applyColorMap(display_blending, cv2.COLORMAP_JET)
            display_blending = (display_blending / 255.).astype(np.float32)
            if img_c == 1:
                cls_display_append = np.stack((cls_display_append.squeeze(-1),)*3, axis=-1)
            cls_display_append = np.hstack([cls_display_append, display_blending])
            display_append = np.vstack([display_append, cls_display_append]) if display_append is not None else cls_display_append.copy()
        if window_size_h is not None:
            display_append = cv2.resize(display_append, ((window_size_h * display_append.shape[1]) // display_append.shape[0], window_size_h))
        cv2.imshow('cam', display_append)
        cv2.waitKey(1)
        

    def fit(self):
        self.model.summary()
        optimizer = tf.keras.optimizers.SGD(lr=self.lr, momentum=self.momentum, nesterov=True)
        if not (os.path.exists('checkpoints') and os.path.exists('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)

        print(f'\ntrain on {len(self.train_image_paths)} samples')
        print(f'validate on {len(self.validation_image_paths)} samples\n')
        iteration_count = 0
        while True:
            for idx, (batch_x, batch_y) in enumerate(self.train_data_generator.flow()):
                lr = self.lr_scheduler.schedule_step_decay(optimizer, iteration_count)
                loss = self.compute_gradient(self.model, optimizer, batch_x, batch_y)
                if self.showcam and iteration_count % 100 == 0:
                    try_count = 0
                    while True:
                        if try_count > len(batch_x):
                            break
                        rnum = random.randint(0, len(batch_x) - 1)
                        if np.all(batch_y[rnum] < 0.3): # if unknown? > no display
                            continue
                        else:
                            new_input_tensor = batch_x[rnum]
                            label_idx = np.argmax(batch_y[rnum]).item()
                            break
                    self.draw_cam(self.model, new_input_tensor, label_idx)
                # self.live_loss_plot.update(loss)
                # self.live_lr_plot.update(lr)
                iteration_count += 1
                print(f'\r[iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='')
                if iteration_count == self.iterations:
                    self.save_model(iteration_count)
                    print('train end successfully')
                    exit(0)

    def save_model(self, iteration_count):
        print(f'iteration count : {iteration_count}')
        if self.validation_data_generator.flow() is None:
            self.model.save(f'checkpoints/model_{iteration_count}_iter.h5', include_optimizer=False)
        else:
            val_acc = self.evaluate_core(unknown_threshold=0.5, validation_data_generator=self.validation_data_generator_one_batch)
            if val_acc > self.max_val_acc:
                self.max_val_acc = val_acc
                self.model.save(f'checkpoints/best_model_{iteration_count}_iter_val_acc_{val_acc:.4f}.h5', include_optimizer=False)
                print(f'[best model saved] {iteration_count} iteration => val_acc: {val_acc:.4f}\n')
            else:
                self.model.save(f'checkpoints/model_{iteration_count}_iter_val_acc_{val_acc:.4f}.h5', include_optimizer=False)

    def evaluate(self, unknown_threshold=0.5):
        self.evaluate_core(unknown_threshold=unknown_threshold, validation_data_generator=self.validation_data_generator_one_batch)

    def evaluate_core(self, unknown_threshold=0.5, validation_data_generator=None):
        @tf.function
        def predict(model, x):
            return model(x, training=False)
        num_classes = self.model.output_shape[1]
        true_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        total_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        true_unknown_count = total_unknown_count = 0
        for batch_x, batch_y in tqdm(validation_data_generator.flow()):
            y = predict(self.model, batch_x)[0]
            # with np.printoptions(precision=2, suppress=True):
            #     print(np.asarray(y))
            max_score_index = np.argmax(y)
            max_score = y[max_score_index]
            if np.sum(batch_y[0]) == 0.0:  # case unknown using zero label
                total_unknown_count += 1
                if max_score < unknown_threshold:
                    true_unknown_count += 1
            else:  # case classification
                true_class_index = np.argmax(batch_y[0])
                total_counts[true_class_index] += 1
                if max_score_index == true_class_index and max_score >= unknown_threshold:
                    true_counts[true_class_index] += 1

        print('\n')
        acc_sum = 0.0
        for i in range(len(total_counts)):
            cur_class_acc = true_counts[i] / (float(total_counts[i]) + 1e-5)
            acc_sum += cur_class_acc
            print(f'[class {i:2d}] acc => {cur_class_acc:.4f}')

        valid_class_count = num_classes
        if total_unknown_count > 0:
            unknown_acc = true_unknown_count / float(total_unknown_count + 1e-5)
            acc_sum += unknown_acc
            valid_class_count += 1
            print(f'[class unknown] acc => {unknown_acc:.4f}')

        acc = acc_sum / valid_class_count
        print(f'sigmoid classifier accuracy with unknown threshold({unknown_threshold:.2f}) : {acc:.4f}')
        return acc

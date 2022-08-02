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
                 show_class_activation_map=False,
                 cam_activation_layer_name='activation_4',
                 last_conv_layer_name='conv2d_6'):
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.max_val_acc = 0.0
        self.show_class_activation_map = show_class_activation_map
        self.cam_activation_layer_name = cam_activation_layer_name
        self.last_conv_layer_name = last_conv_layer_name

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

    def draw_cam(self, model, x, label, window_size_h=512, alpha=0.5):
        target_fmap = model.get_layer(name=self.cam_activation_layer_name).output
        activation_h, activation_w, activation_c = target_fmap.shape[1:]
        new_model = tf.keras.Model(self.model.input, target_fmap)
        weights = model.get_layer(name=self.last_conv_layer_name).get_weights()[0]
        weights = weights.squeeze()
        img_h, img_w, img_c = x.shape

        fmap = new_model(x[tf.newaxis, ...], training=False)[0]
        if img_c == 1:
            x = np.concatenate([x, x, x], axis=-1)
        image_grid = None
        for idx, cls in enumerate(self.class_names):
            org_image = x.copy()
            if img_c == 3:
                org_image = cv2.cvtColor(org_image, cv2.COLOR_RGB2BGR)

            class_weights = weights[:, idx]
            cam = np.zeros((activation_h, activation_w), dtype=np.float32)
            for i in range(activation_c):
                cam += class_weights[i] * fmap[:, :, i]
            cam = np.array(cam)

            cam -= np.min(cam)
            cam /= np.max(cam)
            cam *= 255.0
            cam = cam.astype('uint8')
            cam = cv2.resize(cam, (img_w, img_h))
            cam = cam[..., np.newaxis]
            cam = np.concatenate([cam, cam, cam], axis=-1)

            cam_jet = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            cam_blended = cv2.addWeighted((org_image * 255).astype(np.uint8), alpha, cam_jet, (1 - alpha), 0)

            label_box = np.zeros((img_h, 20, 3), dtype=np.float32) + float(label == idx)
            label_box = (label_box * 255.0).astype('uint8')
            org_image = (org_image * 255.0).astype('uint8')
            grid_row = np.concatenate([label_box, org_image, cam, cam_jet, cam_blended], axis=1)
            image_grid = np.append(image_grid, grid_row, axis=0) if image_grid is not None else grid_row.copy()
        if window_size_h is not None:
            image_grid = cv2.resize(image_grid, ((window_size_h * image_grid.shape[1]) // image_grid.shape[0], window_size_h))
        cv2.imshow('cam', image_grid)
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
                if self.show_class_activation_map and iteration_count % 100 == 0:
                    try_count = 0
                    while True:
                        if try_count > len(batch_x):
                            break
                        rnum = random.randint(0, len(batch_x) - 1)
                        if np.all(batch_y[rnum] < 0.3):  # skip cam view if unknown data
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

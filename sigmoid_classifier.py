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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import cv2
import random
import warnings
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from model import Model
from eta import ETACalculator
from live_plot import LivePlot
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ace import AdaptiveCrossentropy
from ckpt_manager import CheckpointManager


class SigmoidClassifier(CheckpointManager):
    def __init__(self,
                 train_image_path,
                 validation_image_path,
                 model_name,
                 input_shape,
                 lr,
                 lrf,
                 alpha,
                 gamma,
                 warm_up,
                 momentum,
                 batch_size,
                 iterations,
                 label_smoothing,
                 aug_brightness,
                 aug_contrast,
                 aug_rotate,
                 aug_h_flip,
                 lr_policy='step',
                 checkpoint_interval=0,
                 show_class_activation_map=False,
                 cam_activation_layer_name='cam_activation',
                 last_conv_layer_name='squeeze_conv'):
        super().__init__()
        assert checkpoint_interval == 0 or checkpoint_interval >= 1000
        self.input_shape = input_shape
        self.lr = lr
        self.lrf = lrf
        self.warm_up = warm_up
        self.alpha = alpha
        self.gamma = gamma
        self.momentum = momentum
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size
        self.iterations = iterations
        self.lr_policy = lr_policy 
        self.show_class_activation_map = show_class_activation_map
        self.cam_activation_layer_name = cam_activation_layer_name
        self.last_conv_layer_name = last_conv_layer_name
        self.checkpoint_interval = checkpoint_interval
        self.pretrained_iteration_count = 0
        warnings.filterwarnings(action='ignore')
        self.set_model_name(model_name)
        if self.checkpoint_interval == 0:
            self.checkpoint_interval = self.iterations

        train_image_path = self.unify_path(train_image_path)
        validation_image_path = self.unify_path(validation_image_path)

        self.train_image_paths, train_class_names, _ = self.init_image_paths(train_image_path)
        self.validation_image_paths, validation_class_names, self.include_unknown = self.init_image_paths(validation_image_path)
        if len(self.train_image_paths) == 0:
            print(f'no images in train_image_path : {train_image_path}')
            exit(0)
        if len(self.validation_image_paths) == 0:
            print(f'no images in validation_image_path : {validation_image_path}')
            exit(0)

        self.class_names = validation_class_names
        self.train_data_generator = DataGenerator(
            root_path=train_image_path,
            image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            class_names=train_class_names,
            aug_brightness=aug_brightness,
            aug_contrast=aug_contrast,
            aug_rotate=aug_rotate,
            aug_h_flip=aug_h_flip)
        self.validation_data_generator = DataGenerator(
            root_path=validation_image_path,
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            class_names=self.class_names)
        self.train_data_generator_one_batch = DataGenerator(
            root_path=train_image_path,
            image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            batch_size=1,
            class_names=train_class_names)
        self.validation_data_generator_one_batch = DataGenerator(
            root_path=validation_image_path,
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=1,
            class_names=self.class_names)

        self.model = Model(
            input_shape=self.input_shape,
            num_classes=len(self.class_names),
            last_conv_layer_name=last_conv_layer_name,
            cam_activation_layer_name=cam_activation_layer_name).build()

    def load_model(self, model_path):
        if os.path.exists(model_path) and os.path.isfile(model_path):
            self.pretrained_iteration_count = self.parse_pretrained_iteration_count(model_path)
            self.model = tf.keras.models.load_model(model_path, compile=False)
        else:
            print(f'pretrained model not found : {model_path}')
            exit(0)

    def unify_path(self, path):
        if path == '':
            return path
        path = path.replace('\\', '/')
        if path.endswith('/'):
            path = path[:-1]
        return path

    def init_image_paths(self, image_path):
        include_unknown = False
        dir_paths = sorted(glob(f'{image_path}/*'))
        for i in range(len(dir_paths)):
            dir_paths[i] = dir_paths[i].replace('\\', '/')
        image_paths = []
        class_counts = []
        class_name_set = set()
        unknown_class_count = 0
        print('class image count')
        for dir_path in dir_paths:
            if not os.path.isdir(dir_path):
                continue
            dir_name = dir_path.split('/')[-1]
            if dir_name[0] == '_':
                print(f'class dir {dir_name} is ignored. dir_name[0] == "_"')
                continue
            if dir_name == 'unknown':
                include_unknown = True
            else:
                class_name_set.add(dir_name)
            cur_class_image_paths = glob(f'{dir_path}/**/*.jpg', recursive=True)
            for i in range(len(cur_class_image_paths)):
                cur_class_image_paths[i] = cur_class_image_paths[i].replace('\\', '/')
            image_paths += cur_class_image_paths
            cur_class_image_count = len(cur_class_image_paths)
            if dir_name == 'unknown':
                unknown_class_count = cur_class_image_count
            else:
                class_counts.append(cur_class_image_count)
            print(f'class {dir_name} : {cur_class_image_count}')
        print()
        class_names = sorted(list(class_name_set))
        total_data_count = float(sum(class_counts)) + unknown_class_count
        return image_paths, class_names, include_unknown

    @tf.function
    def compute_gradient(self, model, optimizer, batch_x, y_true, loss_function):
        with tf.GradientTape() as tape:
            y_pred = self.model(batch_x, training=True)
            loss = tf.reduce_mean(loss_function(y_true, y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def draw_cam(self, x, label, window_size_h=512, alpha=0.6):
        cam_activation_layer = self.model.get_layer(name=self.cam_activation_layer_name).output
        activation_h, activation_w, activation_c = cam_activation_layer.shape[1:]
        cam_model = tf.keras.Model(self.model.input, cam_activation_layer)
        weights = np.asarray(self.model.get_layer(name=self.last_conv_layer_name).get_weights()[0].squeeze())
        img_h, img_w, img_c = x.shape

        activation_map = np.asarray(cam_model(x[tf.newaxis, ...], training=False)[0])
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
                cam += class_weights[i] * activation_map[:, :, i]
            cam = np.array(cam)

            cam -= np.min(cam)
            cam /= np.max(cam)
            cam *= 255.0
            cam = cam.astype(np.uint8)
            cam = cv2.resize(cam, (img_w, img_h))
            cam = cam[..., np.newaxis]
            cam = np.concatenate([cam, cam, cam], axis=-1)

            cam_jet = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            cam_blended = cv2.addWeighted((org_image * 255).astype(np.uint8), alpha, cam_jet, (1 - alpha), 0)

            label_box = np.zeros((img_h, 20, 3), dtype=np.float32) + float(label == idx)
            label_box = (label_box * 255.0).astype(np.uint8)
            org_image = (org_image * 255.0).astype(np.uint8)
            grid_row = np.concatenate([label_box, org_image, cam, cam_jet, cam_blended], axis=1)
            image_grid = np.append(image_grid, grid_row, axis=0) if image_grid is not None else grid_row.copy()
        if window_size_h is not None:
            image_grid = cv2.resize(image_grid, ((window_size_h * image_grid.shape[1]) // image_grid.shape[0], window_size_h))
        cv2.imshow('cam', image_grid)
        cv2.waitKey(1)

    def print_loss(self, progress_str, loss):
        print(f'\r{progress_str} loss => {loss:.4f}', end='')

    def train(self):
        if self.pretrained_iteration_count >= self.iterations:
            print(f'pretrained iteration count {self.pretrained_iteration_count} is greater or equal than target iterations {self.iterations}')
            exit(0)

        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples')
        print(f'validate on {len(self.validation_image_paths)} samples\n')
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.momentum)
        loss_function = AdaptiveCrossentropy(alpha=self.alpha, gamma=self.gamma, label_smoothing=self.label_smoothing)
        lr_scheduler = LRScheduler(lr=self.lr, lrf=self.lrf, iterations=self.iterations, warm_up=self.warm_up, policy=self.lr_policy)
        self.init_checkpoint_dir()
        iteration_count = self.pretrained_iteration_count
        eta_calculator = ETACalculator(iterations=self.iterations, start_iteration=iteration_count)
        eta_calculator.start()
        while True:
            batch_x, batch_y = self.train_data_generator.load()
            lr_scheduler.update(optimizer, iteration_count)
            loss = self.compute_gradient(self.model, optimizer, batch_x, batch_y, loss_function)
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
                self.draw_cam(new_input_tensor, label_idx)
            iteration_count += 1
            progress_str = eta_calculator.update(iteration_count)
            self.print_loss(progress_str, loss)
            if iteration_count % 2000 == 0:
                self.save_last_model(self.model, iteration_count)
            if iteration_count >= int(self.iterations * self.warm_up) and iteration_count % self.checkpoint_interval == 0:
                acc, class_score, unknown_score = self.evaluate()
                content = f'_acc_{acc:.4f}_class_score_{class_score:.4f}'
                if self.include_unknown:
                    content += f'_unknown_score_{unknown_score:.4f}'
                self.save_best_model(self.model, iteration_count, content=content, metric=acc)
            if iteration_count == self.iterations:
                print('\ntrain end successfully')
                break

    def evaluate(self, dataset='validation', unknown_threshold=0.5):
        assert dataset in ['train', 'validation']
        if dataset == 'train':
            data_generator = self.train_data_generator_one_batch
        else:
            data_generator = self.validation_data_generator_one_batch

        @tf.function
        def graph_forward(model, x):
            return model(x, training=False)

        print()
        num_classes = self.model.output_shape[1]
        hit_class_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        total_class_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        hit_class_score_sums = np.zeros(shape=(num_classes,), dtype=np.float32)
        hit_unknown_count = 0
        total_unknown_count = 0
        hit_unknown_score_sum = 0.0
        for _ in tqdm(range(len(data_generator))):
            batch_x, batch_y = data_generator.load()
            y = graph_forward(self.model, batch_x)[0]
            max_score_index = np.argmax(y)
            max_score = y[max_score_index]
            if np.sum(batch_y[0]) == 0.0:  # case unknown using zero label
                total_unknown_count += 1
                if max_score < unknown_threshold:
                    hit_unknown_count += 1
                    hit_unknown_score_sum += max_score
            else:  # case classification
                true_class_index = np.argmax(batch_y[0])
                total_class_counts[true_class_index] += 1
                if max_score_index == true_class_index:
                    if self.include_unknown:
                        if max_score >= unknown_threshold:
                            hit_class_counts[true_class_index] += 1
                            hit_class_score_sums[true_class_index] += max_score
                    else:
                        hit_class_counts[true_class_index] += 1
                        hit_class_score_sums[true_class_index] += max_score

        total_acc_sum = 0.0
        class_score_sum = 0.0
        for i in range(len(total_class_counts)):
            cur_class_acc = hit_class_counts[i] / (float(total_class_counts[i]) + 1e-5)
            cur_class_score = hit_class_score_sums[i] / (float(hit_class_counts[i]) + 1e-5)
            total_acc_sum += cur_class_acc
            class_score_sum += cur_class_score
            print(f'[class {i:2d}] acc : {cur_class_acc:.4f}, score : {cur_class_score:.4f}')

        valid_class_count = num_classes
        unknown_score = 0.0
        if self.include_unknown and total_unknown_count > 0:
            unknown_acc = hit_unknown_count / float(total_unknown_count + 1e-5)
            unknown_score = hit_unknown_score_sum / float(hit_unknown_count + 1e-5)
            total_acc_sum += unknown_acc
            valid_class_count += 1
            print(f'[class unknown] acc : {unknown_acc:.4f}, score : {unknown_score:.4f}')

        class_acc = total_acc_sum / valid_class_count
        class_score = class_score_sum / num_classes
        if self.include_unknown:
            print(f'total accuracy with unknown threshold({unknown_threshold:.2f}) : {class_acc:.4f}, class_score : {class_score:.4f}, unknown_score : {unknown_score:.4f}\n')
        else:
            print(f'total accuracy : {class_acc:.4f}, class_score : {class_score:.4f}\n')
        return class_acc, class_score, unknown_score


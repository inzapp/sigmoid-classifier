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
import warnings
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from model import Model
from eta import ETACalculator
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
                 checkpoint_interval=0):
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
            num_classes=len(self.class_names)).build()

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

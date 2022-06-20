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

from generator import SigmoidClassifierDataGenerator
from live_loss_plot import LiveLossPlot
from model import Model


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
                 validation_split=0.2):
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.max_val_acc = 0.0

        train_image_path = self.unify_path(train_image_path)
        validation_image_path = self.unify_path(validation_image_path)

        if validation_image_path != '':
            self.train_image_paths, _, self.class_names = self.init_image_paths(train_image_path)
            self.validation_image_paths, _, self.class_names = self.init_image_paths(validation_image_path)
        else:
            self.train_image_paths, self.validation_image_paths, self.class_names = self.init_image_paths(train_image_path, validation_split)

        self.train_data_generator = SigmoidClassifierDataGenerator(
            root_path=train_image_path,
            image_paths=self.balance_class(train_image_path, self.train_image_paths),
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            class_names=self.class_names)
        self.validation_data_generator = SigmoidClassifierDataGenerator(
            root_path=validation_image_path,
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            class_names=self.class_names)
        self.validation_data_generator_one_batch = SigmoidClassifierDataGenerator(
            root_path=train_image_path if validation_image_path == '' else validation_image_path,
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=1,
            class_names=self.class_names)

        if pretrained_model_path != '':
            self.model = tf.keras.models.load_model(pretrained_model_path, compile=False)
        else:
            self.model = Model(input_shape=self.input_shape, num_classes=len(self.class_names)).build()
            self.model.save('model.h5', include_optimizer=False)
        self.live_loss_plot = LiveLossPlot()

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
    def compute_gradient(self, model, optimizer, batch_x, y_true, lr):
        with tf.GradientTape() as tape:
            y_pred = self.model(batch_x, training=True)
            loss = K.binary_crossentropy(y_true, y_pred)
            loss = tf.reduce_mean(loss, axis=0)
            mean_loss = tf.reduce_mean(loss)
            gradients = tape.gradient(loss * lr, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mean_loss

    def fit(self):
        self.model.summary()
        optimizer = tf.keras.optimizers.SGD(lr=1.0, momentum=self.momentum, nesterov=True)
        if not (os.path.exists('checkpoints') and os.path.exists('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)

        print(f'\ntrain on {len(self.train_image_paths)} samples')
        print(f'validate on {len(self.validation_image_paths)} samples\n')
        iteration_count = 0
        while True:
            for batch_x, batch_y in self.train_data_generator.flow():
                lr = self.lr * pow(iteration_count / 1000.0, 4) if iteration_count < 1000 else self.lr
                loss = self.compute_gradient(self.model, optimizer, batch_x, batch_y, tf.constant(lr))
                self.live_loss_plot.update(loss)
                iteration_count += 1
                print(f'\r[iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='')
                if iteration_count % 2000 == 0:
                    self.save_model(iteration_count)
                if iteration_count == self.iterations:
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

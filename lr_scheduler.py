
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
import numpy as np


class LRScheduler:
    def __init__(self,
                 iterations,
                 lr=0.001,
                 min_lr=0.0,
                 min_momentum=0.85,
                 max_momentum=0.95,
                 initial_cycle_length=2500,
                 cycle_weight=2):
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = self.lr
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.iterations = iterations
        self.cycle_length = initial_cycle_length
        self.cycle_weight = cycle_weight
        self.cycle_step = 0

    def __set_lr(self, optimizer, lr):
        optimizer.__setattr__('lr', lr)

    def __set_momentum(self, optimizer, momentum):
        attr = ''
        if optimizer.__str__().lower().find('sgd') > -1:
            attr = 'momentum'
        elif optimizer.__str__().lower().find('adam') > -1:
            attr = 'beta_1'
        if attr != '':
            optimizer.__setattr__(attr, momentum)
        else:
            print(f'__set_momentum() failure. sgd and adam is available optimizers only.')

    def schedule_step_decay(self, optimizer, iteration_count, burn_in=1000):
        if iteration_count <= burn_in:
            lr = self.lr * pow(iteration_count / float(burn_in), 4)
        elif iteration_count == int(self.iterations * 0.8):
            lr = self.lr * 0.1
        else:
            lr = self.lr
        self.__set_lr(optimizer, lr)
        return lr

    def schedule_one_cycle(self, optimizer, iteration_count):
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(((1.0 / (self.iterations * 0.5)) * np.pi * iteration_count) + np.pi))  # up and down
        self.__set_lr(optimizer, lr)
        momentum = self.min_momentum + 0.5 * (self.max_momentum - self.min_momentum) * (1.0 + np.cos(((1.0 / (self.iterations * 0.5)) * np.pi * (iteration_count % self.iterations))))  # down and up
        self.__set_momentum(optimizer, momentum)
        return lr

    def schedule_cosine_warm_restart(self, optimizer, iteration_count, burn_in=1000):
        if iteration_count <= burn_in:
            lr = self.lr * pow(iteration_count / float(burn_in), 4)
        else:
            if self.cycle_step % self.cycle_length == 0 and self.cycle_step != 0:
                self.cycle_step = 0
                self.cycle_length = int(self.cycle_length * self.cycle_weight)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(((1.0 / self.cycle_length) * np.pi * (self.cycle_step % self.cycle_length))))  # down and down
            self.cycle_step += 1
        self.__set_lr(optimizer, lr)
        return lr


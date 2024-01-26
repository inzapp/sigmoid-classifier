"""
Authors : inzapp

Github url : https://github.com/inzapp/lr-scheduler

Copyright (c) 2022 Inzapp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


class LRScheduler:
    def __init__(self,
                 iterations,
                 lr,
                 policy,
                 warm_up=0.1,
                 min_momentum=0.85,
                 max_momentum=0.95,
                 initial_cycle_length=2500,
                 cycle_weight=2,
                 decay_step=0.1,
                 double_step=True):
        assert 0.0 <= lr <= 1.0
        assert 0.0 <= warm_up <= 1.0
        assert 0.0 <= min_momentum <= 1.0
        assert 0.0 <= max_momentum <= 1.0
        assert 0.0 <= decay_step <= 1.0
        assert policy in ['constant', 'step', 'cosine', 'onecycle']
        self.lr = lr
        self.policy = policy
        self.max_lr = self.lr
        self.warm_up = warm_up
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.iterations = iterations
        self.cycle_length = initial_cycle_length
        self.cycle_weight = cycle_weight
        self.decay_step = decay_step 
        self.min_lr = self.lr * self.decay_step ** 2.0
        self.double_step = double_step
        self.cycle_step = 0

    def update(self, optimizer, iteration_count):
        if self.policy == 'step':
            lr = self.__schedule_step_decay(optimizer, iteration_count)
        elif self.policy == 'cosine':
            lr = self.__schedule_cosine_warm_restart(optimizer, iteration_count)
        elif self.policy == 'onecycle':
            lr = self.__schedule_one_cycle(optimizer, iteration_count)
        elif self.policy == 'constant':
            lr = self.lr
        else:
            print(f'{self.policy} is invalid lr policy.')
            lr = None
        return lr

    def __set_lr(self, optimizer, lr):
        optimizer.__setattr__('learning_rate', lr)

    def __set_momentum(self, optimizer, momentum):
        optimizer_str = optimizer.__str__().lower()
        if optimizer_str.find('sgd') > -1:
            optimizer.__setattr__('momentum', momentum)
        elif optimizer_str.find('adam') > -1:
            optimizer.__setattr__('beta_1', momentum)

    def __warm_up_lr(self, iteration_count, warm_up):
        return ((np.cos(((iteration_count * np.pi) / warm_up) + np.pi) + 1.0) * 0.5) * self.lr  # cosine warm up

    def __schedule_step_decay(self, optimizer, iteration_count):
        warm_up_iteration = self.iterations * self.warm_up
        if warm_up_iteration > 0 and iteration_count <= warm_up_iteration:
            lr = self.__warm_up_lr(iteration_count, warm_up_iteration)
        elif self.double_step and iteration_count >= int(self.iterations * 0.9):
            lr = self.lr * self.decay_step ** 2.0
        elif iteration_count >= int(self.iterations * 0.8):
            lr = self.lr * self.decay_step
        else:
            lr = self.lr
        self.__set_lr(optimizer, lr)
        return lr

    def __schedule_one_cycle(self, optimizer, iteration_count):
        min_lr = 0.0
        max_lr = self.max_lr
        min_mm = self.min_momentum
        max_mm = self.max_momentum
        warm_up_iterations = int(self.iterations * self.warm_up)
        if warm_up_iterations > 0 and iteration_count <= warm_up_iterations:
            iterations = warm_up_iterations
            lr = ((np.cos(((iteration_count * np.pi) / iterations) + np.pi) + 1.0) * 0.5) * (max_lr - min_lr) + min_lr  # increase only until target iterations
            mm = ((np.cos(((iteration_count * np.pi) / iterations) +   0.0) + 1.0) * 0.5) * (max_mm - min_mm) + min_mm  # decrease only until target iterations
            self.__set_lr(optimizer, lr)
            self.__set_momentum(optimizer, mm)
        else:
            min_lr = self.min_lr
            iteration_count -= warm_up_iterations + 1
            iterations = self.iterations - warm_up_iterations
            lr = ((np.cos(((iteration_count * np.pi) / iterations) +   0.0) + 1.0) * 0.5) * (max_lr - min_lr) + min_lr  # decrease only until target iterations
            mm = ((np.cos(((iteration_count * np.pi) / iterations) + np.pi) + 1.0) * 0.5) * (max_mm - min_mm) + min_mm  # increase only until target iterations
            self.__set_lr(optimizer, lr)
            self.__set_momentum(optimizer, mm)
        return lr

    def __schedule_cosine_warm_restart(self, optimizer, iteration_count):
        warm_up_iteration = self.iterations * self.warm_up
        if warm_up_iteration > 0 and iteration_count <= warm_up_iteration:
            lr = self.__warm_up_lr(iteration_count, warm_up_iteration)
        else:
            if self.cycle_step % self.cycle_length == 0 and self.cycle_step != 0:
                self.cycle_step = 0
                self.cycle_length = int(self.cycle_length * self.cycle_weight)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(((1.0 / self.cycle_length) * np.pi * (self.cycle_step % self.cycle_length))))  # down and down
            self.cycle_step += 1
        self.__set_lr(optimizer, lr)
        return lr


def plot_lr(policy):
    import tensorflow as tf
    from matplotlib import pyplot as plt
    lr = 0.001
    warm_up = 0.3
    decay_step = 0.2
    iterations = 37500
    iterations = int(iterations / (1.0 - warm_up))
    optimizer = tf.keras.optimizers.SGD()
    lr_scheduler = LRScheduler(iterations=iterations, lr=lr, warm_up=warm_up, decay_step=decay_step, policy=policy, double_step=True)
    lrs = []
    for i in range(iterations):
        lr = lr_scheduler.update(optimizer=optimizer, iteration_count=i)
        lrs.append(lr)
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.legend(['lr'])
    plt.xlabel('iterations')
    plt.tight_layout(pad=0.5)
    plt.show()
    

if __name__ == '__main__':
    plot_lr('constant')
    plot_lr('step')
    plot_lr('onecycle')
    plot_lr('cosine')


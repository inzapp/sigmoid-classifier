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
from matplotlib import pyplot as plt


class LivePlot:
    def __init__(self, iterations, interval=20, mean=10, y_min=0.0, y_max=0.2, legend='loss'):
        plt.style.use(['dark_background'])
        self.fig, self.ax = plt.subplots()
        pad = ((y_max - y_min) * 0.05)
        self.interval = interval
        self.mean = mean
        self.y_min = y_min - pad
        self.y_max = y_max + pad
        self.ax.set_ylim(self.y_min, self.y_max)
        self.data = np.array([None for _ in range(iterations)], dtype=np.float32)
        self.values, = self.ax.plot(np.random.rand(iterations))
        self.recent_values = []
        self.interval_count = 0
        self.index = 0
        plt.xlabel('Iteration')
        plt.legend([legend])
        plt.tight_layout(pad=0.5)

    def update(self, val):
        if val < self.y_min:
            val = self.y_min * 0.99
        elif val > self.y_max:
            val = self.y_max * 0.99
        self.data[self.index] = self.get_recent_avg_value(val)
        self.index += 1
        self.interval_count += 1
        if self.interval_count == self.interval:
            self.interval_count = 0
            self.values.set_ydata(self.data)
            plt.pause(1e-9)

    def get_recent_avg_value(self, val):
        if len(self.recent_values) > self.mean:
            self.recent_values.pop(0)
        self.recent_values.append(val)
        return np.mean(self.recent_values)


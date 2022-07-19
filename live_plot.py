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


class LivePlot():
    def __init__(self, batch_range=10000, range_min=-0.05, range_max=1.0, title='Live plot', legend='Loss'):
        super().__init__()
        plt.style.use(['dark_background'])
        self.fig, self.ax = plt.subplots()
        self.range_min = range_min
        self.range_max = range_max
        self.ax.set_ylim(self.range_min, self.range_max)
        self.data = [np.NaN for _ in range(batch_range)]
        self.values, = self.ax.plot(np.random.rand(batch_range))
        self.recent_values = []
        self.skip_count = 0
        plt.gcf().canvas.set_window_title(title)
        plt.xlabel('Batch range')
        plt.legend([legend])
        plt.tight_layout(pad=0.5)

    def update(self, loss, skip_count=20):
        if loss < self.range_min:
            loss = self.range_min * 0.99
        elif loss > self.range_max:
            loss = self.range_max * 0.99
        self.data.pop(0)
        self.data.append(self.get_recent_avg_value(loss))
        self.skip_count += 1
        if self.skip_count == skip_count:
            self.skip_count = 0
            self.values.set_ydata(self.data)
            plt.pause(1e-9)

    def get_recent_avg_value(self, loss):
        if len(self.recent_values) > 10:
            self.recent_values.pop(0)
        self.recent_values.append(loss)
        return np.mean(self.recent_values)

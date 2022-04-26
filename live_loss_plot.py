import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class LiveLossPlot(tf.keras.callbacks.Callback):
    def __init__(self, batch_range=5000, range_min=-0.05, range_max=1.0):
        super().__init__()
        plt.style.use(['dark_background'])
        self.fig, self.ax = plt.subplots()
        self.range_min = range_min
        self.range_max = range_max
        self.ax.set_ylim(self.range_min, self.range_max)
        self.data = [np.NaN for _ in range(batch_range)]
        self.values, = self.ax.plot(np.random.rand(batch_range))
        self.recent_values = []
        plt.gcf().canvas.set_window_title('Live loss plot')
        plt.xlabel('Batch range')
        plt.legend(['Loss'])
        plt.tight_layout(pad=0.5)

    def on_batch_end(self, batch, loss=None):
        self.update(loss)

    def update(self, loss):
        if loss < self.range_min:
            loss = self.range_min * 0.99
        elif loss > self.range_max:
            loss = self.range_max * 0.99
        self.data.pop(0)
        self.data.append(self.get_recent_avg_value(loss))
        self.values.set_ydata(self.data)
        plt.pause(1e-9)

    def get_recent_avg_value(self, loss):
        if len(self.recent_values) > 10:
            self.recent_values.pop(0)
        self.recent_values.append(loss)
        return np.mean(self.recent_values)

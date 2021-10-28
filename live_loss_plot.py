import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class LiveLossPlot(tf.keras.callbacks.Callback):
    def __init__(self, batch_range=5000):
        super().__init__()
        plt.style.use(['dark_background'])
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-0.05, 1.0)
        self.data = [np.NaN for _ in range(batch_range)]
        self.values, = self.ax.plot(np.random.rand(batch_range))
        self.recent_values = []
        plt.gcf().canvas.set_window_title('Live loss plot')
        plt.xlabel('Batch range')
        plt.legend(['Loss'])
        plt.tight_layout(pad=0.5)

    def on_batch_end(self, batch, logs=None):
        self.update(logs)

    def update(self, logs):
        self.data.pop(0)
        self.data.append(self.get_recent_avg_value(logs))
        self.values.set_ydata(self.data)
        plt.pause(1e-9)

    def get_recent_avg_value(self, logs):
        if len(self.recent_values) > 10:
            self.recent_values.pop(0)
        self.recent_values.append(logs['loss'])
        return np.mean(self.recent_values)

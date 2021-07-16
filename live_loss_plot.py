import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class LiveLossPlot(tf.keras.callbacks.Callback):
    def __init__(self, x_range=1000):
        super().__init__()
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-0.1, 1.0)
        self.data = [None for _ in range(x_range)]
        self.values, = self.ax.plot(np.random.rand(x_range))
        plt.xlabel('Batch range')
        plt.legend(['Loss'])
        plt.tight_layout(pad=1.0)
        plt.gcf().canvas.set_window_title('Live loss plot')

    def update(self, logs):
        self.data.pop(0)
        self.data.append(logs['loss'])
        self.values.set_ydata(self.data)
        plt.pause(1e-9)

    def on_batch_end(self, batch, logs=None):
        self.update(logs)

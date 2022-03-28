from typing import Callable

import tensorflow as tf
from tensorflow import keras


class Model(keras.Model):

    def __init__(self, units):
        super().__init__()
        self.dense1 = keras.layers.Dense(
            units=units,
            activation=tf.nn.relu,
            kernel_initializer=tf.random.normal,
            bias_initializer=tf.random.normal,
        )
        self.dense2 = keras.layers.Dense(units/2)
        self.dense3 = keras.layers.Dense(units/4)
        self.dense4 = keras.layers.Dense(1)

    def call(self, x2, training=True, mask=None):
        x1 = self.dense1(x2)
        x1 = self.dense2(x1)
        x1 = self.dense3(x1)
        x1 = self.dense4(x1)
        tf.squeeze(x1, axis=-1)
        return x1

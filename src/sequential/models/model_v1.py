import typing

import tensorflow as tf
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class Model(keras.layers.Layer):

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
        self.dense4 = keras.layers.Dense(1, activation=keras.activations.sigmoid)

    def call(self, x2, training=True, mask=None):
        x1 = self.dense1(x2)
        x1 = self.dense2(x1)
        x1 = self.dense3(x1)
        x1 = self.dense4(x1)
        tf.squeeze(x1, axis=-1)
        return x1

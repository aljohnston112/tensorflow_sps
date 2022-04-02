import typing

import matplotlib.pyplot as plt
import tensorflow
import tensorflow as tf
from keras.layers import Layer
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import pi

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class PoissonNeuron(Layer):
    def __init__(self, num_output_to_input_ratio, **kwargs):
        super().__init__(**kwargs)
        self.b = None
        self.a = None
        self.front = tf.divide(1.0, pi)
        self.num_output_to_input_ratio = num_output_to_input_ratio

    def build(self, input_shape):
        self.a = self.add_weight("a", shape=[self.num_output_to_input_ratio, int(input_shape[-1])])
        self.b = self.add_weight("b", shape=[self.num_output_to_input_ratio, int(input_shape[-1])])

    def call(self, inputs):
        c = tf.divide(tf.subtract(inputs, self.a), self.b)
        c2 = tf.pow(c, 2.0)
        front = tf.multiply(tf.divide(1.0, tf.sqrt(tf.abs(self.b))), self.front)
        top = tf.subtract(1.0, c2)
        bottom = tf.pow(tf.add(1.0, c2), 2.0)
        back = tensorflow.divide(top, bottom)
        o = tf.multiply(front, back)
        return o


class PoissonLayer(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.kernel = None
        self.poisson = None
        self.b = None
        self.a = None
        self.front = tf.divide(1.0, pi)
        self.num_outputs = units

    def build(self, input_shape):
        self.poisson = PoissonNeuron(num_output_to_input_ratio=1, input_shape=input_shape)
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])
        self.a = self.add_weight("a", shape=[self.num_outputs])
        self.b = self.add_weight("b", shape=[self.num_outputs])

    def call(self, inputs):
        p_out = self.poisson(inputs)
        p_in = tf.matmul(p_out, self.kernel)
        c = tf.divide(tf.subtract(p_in, self.a), self.b)
        c2 = tf.pow(c, 2.0)
        front = tf.multiply(tf.divide(1.0, tf.sqrt(tf.abs(self.b))), self.front)
        top = tf.subtract(1.0, c2)
        bottom = tf.pow(tf.add(1.0, c2), 2.0)
        back = tensorflow.divide(top, bottom)
        o = tf.multiply(front, back)
        return o


if __name__ == "__poisson_layer__":
    layer = PoissonLayer(1)
    features = tf.linspace(-1, 1, 100)
    labels = tf.constant(100)

    optimizer = tf.optimizers.Adam(
                tf.keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=0.00001,
                    first_decay_steps=1000
                ))
    loss_fn = keras.losses.MeanSquaredError()

    for _ in range(10000):
        with tf.GradientTape() as tape:
            logits = layer(features)  # Logits for this minibatch
            # Loss value for this minibatch
            loss_value = loss_fn(labels, logits)
            w = layer.get_weights()
            print(loss_value)
        grads = tape.gradient(loss_value, layer.trainable_weights)
        optimizer.apply_gradients(zip(grads, layer.trainable_weights))

    fig, ax = plt.subplots()
    ax.plot(features, layer(features)[0])
    ax.set_xlabel("Features")
    ax.set_ylabel("Layer output")
    ax.set_title("The Poisson Wavelet")
    plt.show()


import typing

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

import matplotlib
from matplotlib import pyplot as plt

import files_prepper
from config import data_root

np.set_printoptions(precision=4)


LABEL_KEY = "Sign"
data_folder = data_root + "dataset_csv/"


def get_single_data_datch(csv_filepaths):
    features = []
    labels = []
    i = 0
    for path in csv_filepaths:
        single_train = pd.read_csv(path)
        single_features = single_train.copy()
        single_labels = single_features.pop(LABEL_KEY).to_frame()
        single_labels.drop(0, inplace=True)
        single_labels.reset_index(drop=True, inplace=True)
        single_features.drop(len(single_features)-1, inplace=True)
        single_features.reset_index(drop=True, inplace=True)
        features.append(single_features)
        labels.append(single_labels)
        i += 1
        if i > 5:
            break
    return features, labels


def normalize_tensors(features):
    layer = keras.layers.Normalization()
    layer.adapt(features)
    return layer(features)


class Model(keras.Model):

    def __init__(self, units):
        super().__init__()
        self.dense1 = keras.layers.Dense(
            units=units,
            activation=tf.nn.relu,
            kernel_initializer=tf.random.normal,
            bias_initializer=tf.random.normal,
        )
        self.dense2 = keras.layers.Dense(1)

    def call(self, x2, training=True, mask=None):
        x1 = self.dense1(x2)
        x1 = self.dense2(x1)
        tf.squeeze(x1, axis=-1)
        return x1


def train_model(model, epochs, learning_rate, features, labels):
    print(f"Number of features in batch: {len(features)}")
    variables = model.variables
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
    for step in range(epochs):
        for i, featu in enumerate(stock_features):
            with tf.GradientTape() as tape:
                prediction = model(featu)
                sub = tf.subtract(labels[i], prediction)
                error = tf.pow(sub, tf.constant(2, dtype=prediction.dtype))
                mean_error = tf.reduce_mean(error)
                if mean_error < 0.0001:
                    break
                gradient = tape.gradient(mean_error, variables)
                optimizer.apply_gradients(zip(gradient, variables))
            if step % 10 == 0:
                print(f'Mean squared error: {mean_error.numpy():0.3f}')


training_filepaths, testing_filepaths, validation_filepaths = files_prepper.random_split(data_folder)
stock_features, stock_labels = get_single_data_datch(training_filepaths)
for k, feat in enumerate(stock_features):
    stock_features[k] = normalize_tensors(feat)


stock_model = Model(64)
stock_model.compile()


# W, H in inches
matplotlib.rcParams['figure.figsize'] = [9, 6]
for j, feat in enumerate(stock_features):
    x = tf.linspace(0, 1, feat.shape[0])
    x = tf.cast(x, tf.float32)
    plt.figure(j)
    plt.plot(x.numpy(), feat.numpy(), '.', label=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    plt.plot(x.numpy(), stock_labels[j].to_numpy(), '*', label="Sign")
    plt.plot(x, stock_model(feat), label='Untrained predictions')
    plt.title('Before training')
    plt.legend()
    plt.draw()
    plt.pause(0.1)


train_model(stock_model, epochs=1000, learning_rate=0.01, features=stock_features, labels=stock_labels)
for k, feat in enumerate(stock_features):
    x = tf.linspace(0, 1, feat.shape[0])
    x = tf.cast(x, tf.float32)
    out = stock_model(feat)
    out = [1 if o > 0.5 else 0for o in out]
    plt.figure(k)
    plt.plot(x, out, label='Trained predictions')
    plt.title('After training')
    plt.legend()
    plt.draw()
    plt.pause(0.1)

plt.show(block=True)

# single_step_window = WindowGenerator(
#     input_width=1, label_width=1, shift=1,
#
#     label_columns=['T (degC)'], )
#
# print(single_step_window)

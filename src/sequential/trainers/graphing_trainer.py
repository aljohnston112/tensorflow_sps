import typing
from math import isnan

import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

from src.main import config
from src.data_transformers import files_prepper
from src.main.config import data_folder, LABEL_KEY
from src.main.models import model_v1


class TerminateAtNan(keras.callbacks.TerminateOnNaN):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        if isnan(logs["loss"]):
            print("End epoch {} of training; got log keys: {}".format(epoch, keys))


def train(batcher_type):
    config.init()
    training_filepaths, testing_filepaths, validation_filepaths = files_prepper.random_split(data_folder)
    training_batcher = batcher_type(training_filepaths[:100], label_key=LABEL_KEY)
    validation_batcher = batcher_type(validation_filepaths[:100], label_key=LABEL_KEY)

    stock_features, stock_labels = training_batcher.get_single_data_batch()
    stock_features_validation, stock_labels_validation = validation_batcher.get_single_data_batch()

    stock_model = model_v1.Model(50)
    stock_model.compile(
        loss=tf.keras.losses.MSE,
        optimizer=tf.optimizers.SGD(learning_rate=0.01, clipvalue=1.0)
    )

    for j in range(1000):
        history_plot = []
        val_history_plot = []
        x_plot = []
        predicted_out_plot = []
        val_predicted_out_plot = []
        unpredicted_out_plot = []
        feature_plot = []
        label_plot = []
        i = 0
        while stock_features is not None:
            feature_plot.append(stock_features)
            label_plot.append(stock_labels)
            out = stock_model(stock_features)
            unpredicted_out_plot.append([1 if o > 0.5 else 0 for o in out])

            print(f'x= {stock_features}, y = {stock_labels}')
            stock_features_validation, stock_labels_validation = validation_batcher.get_single_data_batch()
            while stock_features_validation is None:
                stock_features_validation, stock_labels_validation = validation_batcher.get_single_data_batch()
            history = stock_model.fit(
                stock_features, stock_labels,
                validation_data=(stock_features_validation, stock_labels_validation),
                epochs=100,
                verbose=1,
                callbacks=[TerminateAtNan()],
            )

            history_plot.append(history.history['loss'])
            val_history_plot.append(history.history['val_loss'])
            x_plot.append(tf.linspace(i, i + 1, stock_features.shape[0]))
            out = stock_model(stock_features)
            predicted_out_plot.append([1 if o > 0.5 else 0 for o in out])
            out = stock_model(stock_features_validation)
            val_predicted_out_plot.append([1 if o > 0.5 else 0 for o in out])

            stock_features, stock_labels = training_batcher.get_single_data_batch()
            i += 1
        if len(x_plot) > 0:
            # W, H in inches
            matplotlib.rcParams['figure.figsize'] = [9, 6]
            plt.figure(j)
            plt.plot(tf.concat(x_plot, 0).numpy(), tf.concat(label_plot, 0).numpy(), '*', label="Sign")
            # plt.plot(tf.concat(x_plot, 0).numpy(), tf.concat(unpredicted_out_plot, 0).numpy(),
            #          label='Untrained predictions')
            plt.title('Before training')
            plt.legend()
            plt.draw()
            plt.pause(0.1)

            plt.figure(j + 1)
            plt.plot(tf.concat(history_plot, 0).numpy())
            plt.xlabel('Epoch')
            plt.ylim([0, max(plt.ylim())])
            plt.ylabel('Loss [Mean Squared Error]')
            plt.title('Keras training progress')

            plt.figure(j + 1)
            plt.plot(tf.concat(val_history_plot, 0).numpy())
            plt.ylim([0, max(plt.ylim())])
            plt.legend(['Train', 'Val'])
            plt.title('Keras training progress')

            plt.figure(j)
            plt.plot(tf.concat(x_plot, 0).numpy(), tf.concat(predicted_out_plot, 0).numpy(),
                     label='Trained predictions')
            plt.plot(tf.concat(x_plot, 0).numpy(), tf.concat(val_predicted_out_plot, 0).numpy(),
                     label='Val Trained predictions')
            plt.title('After training')
            plt.legend()
            plt.draw()
            plt.pause(0.1)
            plt.show(block=True)



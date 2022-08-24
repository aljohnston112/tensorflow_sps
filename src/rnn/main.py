import typing

import matplotlib
import pandas
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from src import config
from src.config import LABEL_KEY, yahoo_data_numbers_folder
from src.data_transformers import files_prepper
from src.data_transformers.batchers.concat_batcher import ConcatBatcher
from src.wavelets.poisson_layer import PoissonLayer

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras
from keras import layers, regularizers


def train(batcher_type):
    config.init()
    batch_size = 1000
    training_filepaths, testing_filepaths, validation_filepaths = files_prepper.random_split(yahoo_data_numbers_folder)
    training_batcher = batcher_type(training_filepaths[:100], label_key=LABEL_KEY, batch_size=batch_size)
    validation_batcher = batcher_type(validation_filepaths[:100], label_key=LABEL_KEY, batch_size=batch_size)

    stock_features, stock_labels = training_batcher.get_single_data_batch()
    stock_features_validation, stock_labels_validation = validation_batcher.get_single_data_batch()

    feature_norm_model = tf.keras.Sequential([layers.BatchNormalization()])
    label_norm_model = tf.keras.Sequential([layers.BatchNormalization()])

    model = keras.Sequential([
        PoissonLayer(1)
    ])

    # model = tf.keras.Sequential([
    #     # time_features = tf.reshape(time_features, [len(labels), 60])
    #     layers.Dense(4096, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(2048, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(1024, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(256, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(32, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(16, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(8, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(4, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(2, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(1, activation='tanh', kernel_regularizer=regularizers.l2(0.001)),
    # ])

    # model = tf.keras.Sequential([
    #     layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=[10, 6]),
    #     layers.Flatten(),
    #     layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(1, activation='hard_sigmoid', kernel_regularizer=regularizers.l2(0.001)),
    # ])

    # model = keras.Sequential([
    #     layers.BatchNormalization(),
    #     layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=[8]),
    #     layers.LSTM(32, return_sequences=True),
    #     layers.Flatten(),
    #     layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(1, activation='hard_sigmoid', kernel_regularizer=regularizers.l2(0.001)),
    # ])

    # model.compile(
    #     loss=tf.keras.losses.MSE,
    #     metrics=[
    #         keras.losses.MeanSquaredError(),
    #     ],
    #     optimizer=tf.optimizers.Adam(
    #         tf.keras.optimizers.schedules.CosineDecayRestarts(
    #             initial_learning_rate=0.001,
    #             first_decay_steps=1000 * 100
    #         ))
    #     ,
    # )

    for j in range(1000):
        history_plot = []
        val_history_plot = []
        accuracy_plot = []
        val_accuracy_plot = []
        x_plot = []
        val_x_plot = []
        predicted_out_plot = []
        val_predicted_out_plot = []
        unpredicted_out_plot = []
        feature_plot = []
        label_plot = []
        i = 0
        while stock_features is not None and stock_features_validation is not None:
            stock_features = feature_norm_model(stock_features)
            stock_labels = tf.squeeze(label_norm_model(tf.expand_dims(stock_labels, axis=0)))
            stock_features_validation = feature_norm_model(stock_features_validation)
            stock_labels_validation = tf.squeeze(label_norm_model(tf.expand_dims(stock_labels_validation, axis=0)))

            feature_plot.append(stock_features)
            label_plot.append(stock_labels)
            out = model(stock_features)
            unpredicted_out_plot.append([1 if o > 0.5 else 0 for o in out])

            print(f'x= {stock_features}, y = {stock_labels}')

            # history = model.fit(
            #     stock_features, stock_labels,
            #     batch_size=stock_features.shape[0],
            #     validation_data=(stock_features_validation, stock_labels_validation),
            #     validation_batch_size=stock_features_validation.shape[0],
            #     epochs=10000,
            #     verbose=1,
            #     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=200)]
            # )

            optimizer = tf.optimizers.Adam(
                tf.keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=10000000.0,
                    first_decay_steps=1000 * 100
                ))
            loss_fn = keras.losses.MeanSquaredError()

            for _ in range(100):
                with tf.GradientTape() as tape:
                    logits = model(stock_features)  # Logits for this minibatch
                    # Loss value for this minibatch
                    loss_value = loss_fn(stock_labels, logits)
                    history_plot.append(loss_value)
                    print(f"Loss: {loss_value}")
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # history_plot.append(history.history['loss'])
            # val_history_plot.append(history.history['val_loss'])
            # accuracy_plot.append(history.history['mean_squared_error'])
            # val_accuracy_plot.append(history.history['val_mean_squared_error'])

            x_plot.append(tf.linspace(i, i + 1, stock_features.shape[0]))
            val_x_plot.append(tf.linspace(i, i + 1, stock_features_validation.shape[0]))
            out = model(stock_features)
            # predicted_out_plot.append([1 if o > 0.5 else 0 for o in out])
            predicted_out_plot.append(out)
            out = model(stock_features_validation)
            val_predicted_out_plot.append([1 if o > 0.5 else 0 for o in out])

            stock_features, stock_labels = training_batcher.get_single_data_batch()
            stock_features_validation, stock_labels_validation = validation_batcher.get_single_data_batch()
            i += 1
        training_filepaths, testing_filepaths, validation_filepaths = files_prepper.random_split(
            yahoo_data_numbers_folder)
        training_batcher = batcher_type(training_filepaths[:100], label_key=LABEL_KEY, batch_size=batch_size)
        validation_batcher = batcher_type(validation_filepaths[:100], label_key=LABEL_KEY, batch_size=batch_size)
        stock_features, stock_labels = training_batcher.get_single_data_batch()
        stock_features_validation, stock_labels_validation = validation_batcher.get_single_data_batch()
        if len(x_plot) > 0:
            # W, H in inches
            matplotlib.rcParams['figure.figsize'] = [9, 6]
            plt.figure(j)
            plt.plot(tf.concat(x_plot, 0).numpy(), tf.concat(label_plot, 0).numpy(), '*', label="Normalized Labels")
            # plt.plot(tf.concat(x_plot, 0).numpy(), tf.concat(unpredicted_out_plot, 0).numpy(),
            #          label='Untrained predictions')
            plt.title('Before training')
            plt.legend()
            plt.draw()
            plt.pause(0.1)

            plt.figure(j + 1)
            plt.plot(tf.concat(history_plot, 0).numpy())
            plt.xlabel('Epoch')
            plt.ylabel('Loss [Mean Squared Error]')
            plt.title('Keras training progress')
            plt.figure(j + 1)
            # plt.plot(tf.concat(val_history_plot, 0).numpy())

            # plt.figure(j + 1)
            # plt.plot(tf.concat(accuracy_plot, 0).numpy())
            # plt.xlabel('Epoch')
            # plt.ylabel('Mean Squared Error')
            # plt.figure(j + 1)
            # plt.plot(tf.concat(val_accuracy_plot, 0).numpy())
            # plt.legend(['Train', 'Val', 'MSE', 'Val MSE'])

            plt.figure(j)
            plt.plot(tf.concat(x_plot, 0).numpy(), tf.concat(predicted_out_plot, 0).numpy(),
                     label='Trained predictions')
            plt.title('After training')
            plt.legend()
            plt.draw()
            plt.pause(0.1)
            plt.show(block=True)


train(ConcatBatcher)

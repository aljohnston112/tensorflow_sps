import copy
import random
import typing

import numpy as np
import tensorflow as tf
from keras.api.keras import layers
from numpy import ndarray
from tensorflow import keras

from src import config
from src.config import yahoo_data_numbers_folder, LABEL_KEY
from src.data_transformers import files_prepper
from src.data_transformers.batchers.concat_batcher import ConcatBatcher

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


def get_number_of_weights(weights):
    number_of_weights = 0
    for i in weights:
        number_of_weights += i.size
    return number_of_weights


def verify_same_number_of_weights(weights1, weights2):
    number_of_weights = get_number_of_weights(weights1)
    number_of_weights2 = get_number_of_weights(weights2)
    if number_of_weights != number_of_weights2:
        raise ValueError("Weights do not have the same number of weights")
    return number_of_weights


def get_row_and_index(weights, index):
    index_const = index
    row = 0
    count = weights[row].size - 1
    while count < index_const:
        index -= weights[row].size
        row += 1
        count += weights[row].size
    return row, index


def parent_child_weights(weights1, weights2):
    parent_weights = []
    child_weights = []
    for w in weights1:
        parent_weights.append(copy.deepcopy(w))
    for w in weights2:
        child_weights.append(copy.deepcopy(w))
    return parent_weights, child_weights


def random_crossover_of_weights(weights1, weights2):
    child_weights = []
    for w in weights1:
        child_weights.append(copy.deepcopy(w))
    rows = min(len(weights1), len(weights2))
    row = 0
    while row < rows:
        child_row = child_weights[row].ravel()
        parent_row = weights2[row].ravel()
        shortest = min(len(child_row), len(parent_row))
        for i in range(shortest):
            parent = random.randint(0, 1)
            if parent:
                child_row[i] = parent_row[i]
        row += 1
    return child_weights


def random_crossover_of_models(model1, model2):
    model_layers = []
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()
    for i in range(0, len(weights1), 2):
        input_shape = weights1[i].shape[0]
        outputs = len(weights1[i + 1])
        model_layers.append(
            layers.Dense(outputs, activation='hard_sigmoid', input_shape=[input_shape])
        )
    model = tf.keras.Sequential(model_layers)
    weights = random_crossover_of_weights(weights1, weights2)
    model.set_weights(weights)
    return model


def crossover_weights(weights1, weights2, n_points):
    number_of_weights1 = get_number_of_weights(weights1)
    number_of_weights2 = get_number_of_weights(weights2)
    number_of_weights = min(number_of_weights1, number_of_weights2)
    indices = []
    for i in range(n_points):
        indices.append(random.randrange(0, number_of_weights))
    indices.sort()
    parent_weights, child_weights = parent_child_weights(weights2, weights1)
    last_child_row = 0
    last_child_index = 0
    parent_row_to_add = []
    last_parent_index = 0
    for w in parent_weights:
        parent_row_to_add.extend(w.ravel())
    for i in indices:
        child_row, child_index = get_row_and_index(child_weights, i)
        if len(indices) % 2 != 0:
            last_child_row = child_row
            last_child_index = child_index
        else:
            while last_child_row != child_row:
                flat_child_row = child_weights[last_child_row].ravel()
                parent_weights_left = len(parent_row_to_add) - last_parent_index
                if parent_weights_left < len(flat_child_row):
                    flat_child_row[last_child_index:last_child_index + parent_weights_left] = \
                        parent_row_to_add[last_parent_index:]
                    last_parent_index += parent_weights_left
                else:
                    flat_child_row[last_child_index:len(flat_child_row)] = \
                        parent_row_to_add[last_parent_index:len(flat_child_row)]
                    last_parent_index += len(flat_child_row)
                last_child_index = 0
                last_child_row += 1
            flat_child_row = child_weights[child_row].ravel()
            flat_child_row[last_child_index:child_index] = \
                parent_row_to_add[last_parent_index:last_parent_index + (child_index - last_child_index)]
    return child_weights


def crossover_models(model1, model2, n_points):
    model_layers = []
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()
    for i in range(0, len(weights1), 2):
        input_shape = weights1[i].shape[0]
        outputs = len(weights1[i + 1])
        model_layers.append(
            layers.Dense(outputs, activation='hard_sigmoid', input_shape=[input_shape])
        )
    model = tf.keras.Sequential(model_layers)
    weights = crossover_weights(weights1, weights2, n_points)
    model.set_weights(weights)
    return model


def mutate_weights(weights, n_mutations):
    new_weights = copy.deepcopy(weights)
    number_of_weights = 0
    a = 0
    b = 0
    for i in new_weights:
        number_of_weights += i.size
        a = min(a, i.min())
        b = max(b, i.max())
    n_mutations = min(number_of_weights, n_mutations)
    for i in range(n_mutations):
        index = random.randrange(0, number_of_weights)
        row, index = get_row_and_index(new_weights, index)
        new_weight = random.uniform(a, b)
        flat_row = new_weights[row].ravel()
        flat_row[index] = new_weight
    return new_weights


def mutate_model(model_in, n_mutations):
    model_layers = []
    weights = model_in.get_weights()
    for i in range(0, len(weights), 2):
        input_shape = weights[i].shape[0]
        outputs = len(weights[i+1])
        model_layers.append(
            layers.Dense(outputs, activation='hard_sigmoid', input_shape=[input_shape])
        )
    model = tf.keras.Sequential(model_layers)
    mutated_weights = mutate_weights(weights, n_mutations)
    model.set_weights(mutated_weights)
    return model


def create(weights):
    lays = []
    for i in range(0, len(weights), 2):
        lays.append(layers.Dense(weights[i].shape[-1], activation='hard_sigmoid', input_shape=[weights[i].shape[0]]))
    model = tf.keras.Sequential(lays)
    model.set_weights(weights)
    return model


def add_layer(model_in, max_neurons_per_layer):
    weights = copy.deepcopy(model_in.get_weights())
    row = random.randint(0, int(len(weights)/2))
    inputs = weights[2*row-1].shape[0]
    if row == int(len(weights)/2):
        inputs = weights[2*row-1].shape[-1]
    elif row == 0:
        inputs = weights[0].shape[0]
    outputs = random.randint(1, max_neurons_per_layer)
    if row == 0:
        weights.insert(0, tf.random.uniform(shape=[inputs, inputs]).numpy())
        weights.insert(1, tf.random.uniform(shape=[inputs]).numpy())
    elif row == int(len(weights)/2):
        weights.append(tf.random.uniform(shape=[inputs, inputs]).numpy())
        weights.append(tf.random.uniform(shape=[inputs]).numpy())
    else:
        weights.insert(2*row, tf.random.uniform(shape=[inputs, outputs]).numpy())
        weights.insert(2*row + 1, tf.random.uniform(shape=[outputs]).numpy())
        weights.insert(2*row + 2, tf.random.uniform(shape=[outputs, inputs]).numpy())
        weights.insert(2*row + 3, tf.random.uniform(shape=[inputs]).numpy())
    model = create(weights)
    return model


def start_model():
    model = keras.Sequential([
        layers.Dense(8, activation='hard_sigmoid', input_shape=[8]),
        layers.Dense(7, activation='hard_sigmoid'),
        layers.Dense(6, activation='hard_sigmoid'),
        layers.Dense(5, activation='hard_sigmoid'),
        layers.Dense(4, activation='hard_sigmoid'),
        layers.Dense(3, activation='hard_sigmoid'),
    ])
    return model


def model_compare(model1, model2, features, labels):
    loss1 = model1.evaluate(features, labels, verbose=0)
    loss2 = model2.evaluate(features, labels, verbose=0)
    if loss1 < loss2:
        return 1
    else:
        return -1


def compile_model(model):
    model.compile(
        loss=keras.losses.MSE,
        metrics=[keras.losses.MeanSquaredError(), ],
        optimizer=tf.optimizers.Adam(
            keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=0.001,
                first_decay_steps=1000 * 100
            )
        ),
    )


def get_first_generation():
    models = []
    for _ in range(100):
        model = start_model()
        compile_model(model)
        models.append(model)
    return models


def next_gen(models):
    for i in range(1, 6):
        for j in range(i):
            m = random_crossover_of_models(models[j], models[i + 1])
            compile_model(m)
            models.append(m)
    for i in range(1, 6):
        for j in range(i):
            m = crossover_models(models[j], models[i + 1], n_points=3)
            compile_model(m)
            models.append(m)
    for _ in range(20):
        m = mutate_model(models[0], 33)
        compile_model(m)
        models.append(m)
    for _ in range(20):
        m = add_layer(models[0], 64)
        compile_model(m)
        models.append(m)


def train_models(models, stock_features, stock_labels):
    for model in models:
        model.fit(
            stock_features, stock_labels,
            batch_size=stock_features.shape[0],
            epochs=100,
            verbose=1,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=200)]
        )


def train(batcher_type):
    config.init()
    batch_size = 1000
    training_filepaths, testing_filepaths, validation_filepaths = files_prepper.random_split(yahoo_data_numbers_folder)
    training_batcher = batcher_type(training_filepaths[:100], label_key=LABEL_KEY, batch_size=batch_size)
    validation_batcher = batcher_type(validation_filepaths[:100], label_key=LABEL_KEY, batch_size=batch_size)

    stock_features, stock_labels = training_batcher.get_single_data_batch()
    stock_features_validation, stock_labels_validation = validation_batcher.get_single_data_batch()

    feature_norm_model = keras.Sequential([layers.BatchNormalization()])
    label_norm_model = keras.Sequential([layers.BatchNormalization()])

    models = get_first_generation()
    gen = 1
    last_loss = 0
    loss = 0
    for _ in range(100):
        models.sort(key=lambda x: x.evaluate(stock_features, stock_labels, verbose=0))
        models = models[:20]
        print(f"Gen {gen}: ")
        loss = models[0].evaluate(stock_features, stock_labels, verbose=0)[0]
        print(f"First: {models[0].evaluate(stock_features, stock_labels, verbose=0)}")
        print(f"Second: {models[1].evaluate(stock_features, stock_labels, verbose=0)}")
        train_models(models, stock_features, stock_labels)
        last_loss = loss
        next_gen(models)
        gen += 1

    # model1 = start_model()
    # model2 = tf.keras.Sequential([
    #     layers.Dense(3, activation='hard_sigmoid'),
    #     layers.Dense(3, activation='hard_sigmoid')
    # ])
    # model1(stock_features)
    # model2(stock_features)
    # while True:
    #     model3 = random_crossover_of_models(model1, model2)
    #     model3 = crossover_models(model2, model1, 1)
    #     model3 = mutate_model(model1, 1)
    #     model3 = add_layer(model1, 10)
    #     w2 = model3.get_weights()
    #     print(w2)


train(ConcatBatcher)

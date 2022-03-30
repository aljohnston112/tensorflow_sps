import copy
import random
import typing

import tensorflow as tf
from keras.api.keras import layers
from tensorflow import keras

from src import config
from src.config import yahoo_data_numbers_folder, LABEL_KEY
from src.data_transformers import files_prepper
from src.data_transformers.batchers.concat_batcher import ConcatBatcher

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


def crossover_weights(weights1, weights2, n_points):
    number_of_weights = 0
    for i in weights1:
        number_of_weights += i.size
    number_of_weights2 = 0
    for i in weights2:
        number_of_weights2 += i.size
    if number_of_weights != number_of_weights2:
        raise ValueError("Weights do not have the same number of weights")

    indices = []
    for i in range(int(round(n_points/2.0)) + 1):
        indices.append(random.randrange(0, number_of_weights))
    indices.sort()

    first_parent = random.randint(0, 1)
    parent_weights = []
    child_weights = []
    if first_parent:
        for w in weights1:
            parent_weights.append(copy.deepcopy(w))
        for w in weights2:
            child_weights.append(copy.deepcopy(w))
    else:
        for w in weights2:
            parent_weights.append(copy.deepcopy(w))
        for w in weights1:
            child_weights.append(copy.deepcopy(w))

    last_row = 0
    last_i = 0
    for i in indices:
        index_const = i
        index = i
        row = 0
        count = parent_weights[row].size - 1
        while count < index_const:
            index -= parent_weights[row].size
            row += 1
            count += parent_weights[row].size

        while last_row != row:
            flat_parent_row = parent_weights[last_row].ravel()
            flat_child_row = child_weights[last_row].ravel()
            flat_child_row[last_i:len(flat_child_row)] = flat_parent_row[last_i:len(flat_child_row)]
            last_row += 1
            last_i = 0

        flat_parent_row = parent_weights[row].ravel()
        flat_child_row = child_weights[row].ravel()
        flat_child_row[last_i:index] = flat_parent_row[last_i:index]
        parent_weights, child_weights = child_weights, parent_weights
    if len(indices) % 2 != 0:
        parent_weights, child_weights = child_weights, parent_weights

    return child_weights


def crossover(model1, model2, n_points):
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


def random_selection_of_weights(weights1, weights2):
    number_of_weights = 0
    for i in weights1:
        number_of_weights += i.size
    number_of_weights2 = 0
    for i in weights2:
        number_of_weights2 += i.size
    if number_of_weights != number_of_weights2:
        raise ValueError("Weights do not have the same number of weights")

    child_weights = []
    for w in weights1:
        child_weights.append(copy.deepcopy(w))

    row = 0
    index = 0
    for _ in range(number_of_weights):
        parent = random.randint(0, 1)
        if parent:
            child_weights[row][index] = weights2[row][index]
        index += 1
        if len(child_weights[row]) <= index:
            index -= len(child_weights[row])
            row += 1

    return child_weights


def random_selection(model1, model2):
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

    weights = random_selection_of_weights(weights1, weights2)
    model.set_weights(weights)


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
        index_const = index
        row = 0
        count = new_weights[row].size - 1
        while count < index_const:
            index -= new_weights[row].size
            row += 1
            count += new_weights[row].size
        new_weight = random.uniform(a, b)
        flat_row = new_weights[row].ravel()
        flat_row[index] = new_weight

    return new_weights


def mutate(model_in, n_mutations):
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
    model = tf.keras.Sequential([
        layers.Dense(3, activation='hard_sigmoid', input_shape=[8])
    ])
    model.set_weights(weights)
    return model


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

    model1 = tf.keras.Sequential([
        layers.Dense(3, activation='hard_sigmoid'),
        layers.Dense(3, activation='hard_sigmoid')
    ])
    model2 = tf.keras.Sequential([
        layers.Dense(3, activation='hard_sigmoid'),
        layers.Dense(3, activation='hard_sigmoid')
    ])
    model1(stock_features)
    model2(stock_features)
    while True:
        model3 = mutate(model1, 1)
        # model3 = crossover(model1, model2, 1)
        w2 = model3.get_weights()
        print(w2)


train(ConcatBatcher)

import copy
import random
import typing

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from tensorflow import keras

from src import config
from src.config import yahoo_data_numbers_folder, LABEL_KEY
from src.data_transformers import files_prepper
from src.data_transformers.batchers.concat_batcher import ConcatBatcher
from src.wavelets.poisson_layer import PoissonLayer

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

list_of_model_classes = [PoissonLayer, Dense]


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
        parent_weights += (copy.deepcopy(w))
    for w in weights2:
        child_weights += (copy.deepcopy(w))
    return parent_weights, child_weights


def random_crossover_of_weights(weights1, weights2):
    number_of_weights1 = get_number_of_weights(weights1)
    number_of_weights2 = get_number_of_weights(weights2)
    child_weights = []
    is_one = True
    if number_of_weights1 <= number_of_weights2:
        for w in weights1:
            child_weights.append(copy.deepcopy(w))
    else:
        is_one = False
        for w in weights2:
            child_weights.append(copy.deepcopy(w))
    flat_child_weights = unravel_weights(child_weights)
    flat_parent_weights = unravel_weights(weights1) if not is_one else unravel_weights(weights2)
    for i in range(len(flat_child_weights)):
        change = random.choice([True, False])
        if change:
            flat_child_weights[i] = flat_parent_weights[i]
    return child_weights


def create_new_layers_from_model(model_in):
    model_layers = []
    for layer in model_in.layers:
        input_shape = layer.input_shape
        outputs = layer.output_shape[-1]
        model_layers.append(
            type(layer)(units=outputs, input_shape=[input_shape[-1]])
        )
    return model_layers


def create_new_layers_from_match(model1, model2, weights):
    number_of_weights1 = get_number_of_weights(model1.get_weights())
    number_of_weights2 = get_number_of_weights(model2.get_weights())
    is_one = True
    if number_of_weights1 > number_of_weights2:
        is_one = False
    else:
        weights1 = model1.get_weights()
        for i, w in enumerate(weights):
            if weights1[i].shape != w.shape:
                is_one = False
    if is_one:
        model_layers = create_new_layers_from_model(model1)
    else:
        model_layers = create_new_layers_from_model(model2)
    return model_layers


def random_crossover_of_models(model1, model2):
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()
    weights = random_crossover_of_weights(weights1, weights2)
    model_layers = create_new_layers_from_match(model1, model2, weights)
    model = tf.keras.Sequential(model_layers)
    model.set_weights(weights)
    return model


def get_sorted_crossover_points(number_of_weights):
    indices = []
    n_points = random.randint(1, round(number_of_weights / 2))
    for i in range(n_points):
        indices.append(random.randrange(0, number_of_weights))
    indices.sort()
    return indices


def unravel_weights(weights):
    weights_unraveled = []
    for w in weights:
        weights_unraveled.extend(w.ravel())
    return weights_unraveled


def crossover_weights(weights1, weights2):
    number_of_weights1 = get_number_of_weights(weights1)
    number_of_weights2 = get_number_of_weights(weights2)
    child_weights = []
    is_one = True
    if number_of_weights1 <= number_of_weights2:
        for w in weights1:
            child_weights.append(copy.deepcopy(w))
    else:
        is_one = False
        for w in weights2:
            child_weights.append(copy.deepcopy(w))
    flat_child_weights = unravel_weights(child_weights)
    flat_parent_weights = unravel_weights(weights1) if not is_one else unravel_weights(weights2)
    indices = get_sorted_crossover_points(len(flat_child_weights))
    last_index = 0
    for crossover_index in indices:
        if crossover_index % 2 != 0:
            flat_child_weights[last_index:crossover_index] = \
                flat_parent_weights[last_index:crossover_index]
        last_index = crossover_index
    return child_weights


def crossover_models(model1, model2):
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()
    weights = crossover_weights(weights1, weights2)
    model_layers = create_new_layers_from_match(model1, model2, weights)
    model = tf.keras.Sequential(model_layers)
    model.set_weights(weights)
    return model


def mutate_weights(weights):
    new_weights = copy.deepcopy(weights)
    number_of_weights = 0
    for weights in new_weights:
        number_of_weights += weights.size

    n_mutations = random.randint(1, number_of_weights)
    for weights in range(number_of_weights):
        index = random.randrange(0, number_of_weights)
        row, index = get_row_and_index(new_weights, index)
        new_weight = random.uniform(-1.0, 1.0)
        flat_row = new_weights[row].ravel()
        flat_row[index] = new_weight
    return new_weights


def mutate_model(model_in):
    weights = model_in.get_weights()
    model_layers = create_new_layers_from_model(model_in)
    model = tf.keras.Sequential(model_layers)
    mutated_weights = mutate_weights(weights)
    model.set_weights(mutated_weights)
    return model


def add_layer(model_in, max_neurons_per_layer):
    model_layers = []
    row = random.randint(0, len(model_in.layers))
    outputs_for_new_layer = random.randint(1, max_neurons_per_layer)
    for i, layer in enumerate(model_in.layers):
        input_shape = layer.input_shape
        outputs = layer.output_shape[-1]
        if i == row:
            model_layers.append(
                random.choice(list_of_model_classes)(units=outputs_for_new_layer, input_shape=[input_shape[-1]])
            )
            model_layers.append(
                random.choice(list_of_model_classes)(units=input_shape[-1], input_shape=[outputs_for_new_layer])
            )
        model_layers.append(
            random.choice(list_of_model_classes)(units=outputs, input_shape=[input_shape[-1]])
        )
    return keras.Sequential(model_layers)


def start_model():
    model = keras.Sequential([
        random.choice(list_of_model_classes)(8, input_shape=[8]),
        random.choice(list_of_model_classes)(7),
        random.choice(list_of_model_classes)(6),
        random.choice(list_of_model_classes)(5),
        random.choice(list_of_model_classes)(4),
        random.choice(list_of_model_classes)(3),
        # layers.Dense(3, activation='hard_sigmoid'),
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
                initial_learning_rate=1.0,
                first_decay_steps=10
            )
        ),
    )


def get_first_generation():
    models = []
    for _ in range(10):
        model = start_model()
        compile_model(model)
        models.append(model)
    return models


def next_gen(models_list):
    pick = random.randint(0, len(models_list) - 1)
    models = models_list[pick]
    for i in range(1, 2):
        for j in range(i):
            m = random_crossover_of_models(models[j], models[i + 1])
            compile_model(m)
            models_list[0].append(m)
    for i in range(1, 2):
        for j in range(i):
            m = crossover_models(models[j], models[i + 1])
            compile_model(m)
            models_list[0].append(m)
    for _ in range(2):
        m = mutate_model(models[0])
        compile_model(m)
        models_list[0].append(m)
    models_list.append([])
    for _ in range(10):
        m = add_layer(models[0], 64)
        compile_model(m)
        models_list[-1].append(m)


def train_models(models, stock_features, stock_labels):
    for model in models:
        model.fit(
            stock_features, stock_labels,
            batch_size=stock_features.shape[0],
            epochs=1000,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=2)]
        )


def train(batcher_type):
    config.init()
    batch_size = 1000
    training_filepaths, testing_filepaths, validation_filepaths = files_prepper.random_split(yahoo_data_numbers_folder)
    training_batcher = batcher_type(training_filepaths[:10], label_key=LABEL_KEY, batch_size=batch_size)
    validation_batcher = batcher_type(validation_filepaths[:10], label_key=LABEL_KEY, batch_size=batch_size)

    x, y = training_batcher.get_single_data_batch()
    x2 = np.absolute(x)
    x /= np.max(x2, axis=0)
    y2 = np.absolute(y)
    y /= np.max(y2, axis=0)
    stock_features_validation, stock_labels_validation = validation_batcher.get_single_data_batch()
    models_list = [get_first_generation()]
    gen = 1
    last_loss = 0
    loss = 0
    for _ in range(100):
        for models in models_list:
            models.sort(key=lambda z: z.evaluate(x, y, verbose=0))
        # for i, models in enumerate(models_list):
        #     models_list[i] = models[:3]
        print(f"Gen {gen}: ")
        best_loss = 999
        losses = []
        for i, models in enumerate(models_list):
            loss1 = models[0].evaluate(x, y, verbose=0)
            losses.append(loss1[0])
            best_loss = min(loss1[0], best_loss)
            print(f"First for size gen {i}: {loss1}")
        print(f"Best loss: {best_loss}")
        found = False
        for i, loss in enumerate(losses):
            if loss == best_loss and not found:
                found = True
            elif loss == best_loss and found:
                del models_list[i]
        for models in models_list:
            train_models(models, x, y)
        next_gen(models_list)
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

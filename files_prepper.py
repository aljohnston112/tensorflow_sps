import os
import pathlib

import numpy as np
from numpy.random import shuffle
from sklearn.model_selection import train_test_split


def __get_data_filepaths__(data_folder):
    filepaths = []
    for subdir, dirs, files in os.walk(data_folder):
        for file in files:
            filepaths.append(data_folder + file)
    print(f"Found {len(filepaths)} data files")
    return filepaths


def __shuffle_filepaths__(filepaths):
    shuffle(filepaths)
    return filepaths


def __split_filepaths__(filepaths):
    training_filepaths, rem = train_test_split(np.array(filepaths), train_size=0.8)
    validation_filepaths, testing_filepaths = train_test_split(rem, test_size=0.5)

    print("Training data shape: " + str(training_filepaths.shape))
    print("Testing data shape: " + str(testing_filepaths.shape))
    print("Validation data shape: " + str(validation_filepaths.shape))

    return training_filepaths, testing_filepaths, validation_filepaths


def random_split(data_folder):
    return __split_filepaths__(__shuffle_filepaths__(__get_data_filepaths__(data_folder)))




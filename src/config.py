import numpy as np
from matplotlib import pyplot as plt

data_root = "/home/master/Documents/Programming/TensorFlow_Project/"
data_folder = data_root + "dataset_csv/"

LABEL_KEY = "Sign"


def init():
    np.set_printoptions(precision=4)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

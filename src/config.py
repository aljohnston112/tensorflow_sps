import numpy as np

project_root = "/home/master/Documents/Programming/TensorFlow_Project/"
yahoo_data_folder = project_root + "yahoo-data/"
yahoo_data_percent_folder = project_root + "yahoo-data-percents/"
yahoo_data_direction_folder = project_root + "yahoo-data-directions/"
yahoo_data_dataset_folder = project_root + "yahoo-data-datasets/"
yahoo_data_numbers_folder = project_root + "yahoo-data-numbers/"
yahoo_data_aggregate_folder = project_root + "yahoo-data-aggregate"

LABEL_KEY = "Profit"


def init():
    np.set_printoptions(precision=4)

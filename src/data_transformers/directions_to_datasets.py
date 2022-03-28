import glob
import os

import pandas
import pandas as pd
import tensorflow as tf

from src.config import project_root, yahoo_data_folder

labels = project_root + "yahoo-data-signs/"


def convert_files():
    files = [f for f in glob.glob(yahoo_data_folder + "*.csv")]
    out_folder = project_root + "dataset_csv/"
    for feature_file in files:
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        with open(feature_file, 'r') as f:
            out_file = out_folder + os.path.basename(f.name)
            if not os.path.exists(out_file):
                label_file = labels + os.path.basename(f.name)
                fs = pandas.read_csv(f).fillna(0)
                del fs["Date"]
                ls = pandas.read_csv(label_file)
                ls.columns = ["Sign"]
                with open(out_file, "w") as f2:
                    o = pandas.concat([ls, fs], axis=1)
                    f2.write(o.to_csv(index=False))


convert_files()


def get_data_from_csv_path(filepath, label_key):
    single_data = pd.read_csv(filepath)
    single_features = single_data.copy()
    single_labels = single_features.pop(label_key).to_frame()
    single_labels.drop(0, inplace=True)
    single_labels.reset_index(drop=True, inplace=True)
    single_labels = tf.cast(single_labels, tf.float32)
    single_features.drop(len(single_features) - 1, inplace=True)
    single_features.reset_index(drop=True, inplace=True)
    try:
        tf.debugging.check_numerics(single_features, message="")
        tf.debugging.check_numerics(single_labels, message="")
    except Exception as e:
        print(f"nan found in {filepath}")
    return single_features, single_labels

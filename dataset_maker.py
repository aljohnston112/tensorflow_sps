import glob
import os

import pandas

from config import data_root

features = data_root + "yahoo-data/"
labels = data_root + "yahoo-data-signs/"


def convert_files():
    files = [f for f in glob.glob(features + "*.csv")]
    out_folder = data_root + "dataset_csv/"
    for feature_file in files:
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        with open(feature_file, 'r') as f:
            out_file = out_folder + os.path.basename(f.name)
            if not os.path.exists(out_file):
                label_file = labels + os.path.basename(f.name)
                fs = pandas.read_csv(f)
                del fs["Date"]
                ls = pandas.read_csv(label_file)
                ls.columns = ["Sign"]
                with open(out_file, "w") as f2:
                    o = pandas.concat([ls, fs], axis=1)
                    f2.write(o.to_csv(index=False))



convert_files()

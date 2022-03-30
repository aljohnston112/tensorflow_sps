import glob
import os

import pandas

from src.config import yahoo_data_percent_folder, yahoo_data_direction_folder


def convert_percents_to_directions():
    files = [f for f in glob.glob(yahoo_data_percent_folder + "*.csv")]
    out_folder = yahoo_data_direction_folder
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    for percent_file in files:
        with open(percent_file, 'r') as f:
            direction_file = out_folder + os.path.basename(f.name)
            if not os.path.exists(direction_file):
                raw_data = pandas.read_csv(f, index_col=[0])
                direction_data = raw_data.applymap(lambda d: 1 if d > 0 else 0)
                with open(direction_file, "w") as f2:
                    f2.write(direction_data.to_csv())


convert_percents_to_directions()

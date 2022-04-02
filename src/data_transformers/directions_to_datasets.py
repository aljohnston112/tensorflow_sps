import glob
import os

import pandas

from src.config import yahoo_data_direction_folder, yahoo_data_dataset_folder


def convert_files():
    files = [f for f in glob.glob(yahoo_data_direction_folder + "*.csv")]
    out_folder = yahoo_data_dataset_folder
    for direction_file in files:
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        with open(direction_file, 'r') as f:
            out_file = out_folder + os.path.basename(f.name)
            if not os.path.exists(out_file):
                direction_data = pandas.read_csv(f, index_col=[0])

                profit_data = direction_data.pop("Profit")
                direction_data.drop(direction_data.tail(1).index, inplace=True)
                profit_data.drop(profit_data.index[:1], inplace=True)
                direction_data.reset_index(inplace=True, drop=True)
                profit_data.reset_index(inplace=True, drop=True)
                direction_data["Profit"] = profit_data
                with open(out_file, "w") as f2:
                    f2.write(direction_data.to_csv())


convert_files()

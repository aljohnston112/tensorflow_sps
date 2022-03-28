import glob
import os

import pandas

from src.config import project_root, yahoo_data_folder, yahoo_data_percent_folder


def convert_data_to_percents():
    files = [f for f in glob.glob(yahoo_data_folder + "*.csv")]
    out_folder = project_root + yahoo_data_percent_folder
    for data_file in files:
        with open(data_file, 'r') as f:
            percent_file = out_folder + os.path.basename(f.name)
            if not os.path.exists(percent_file):
                raw_data = pandas.read_csv(f)
                raw_data.pop("Date")
                percent_data = raw_data.pct_change()
                percent_data.drop(percent_data.index[:1], inplace=True)
                with open(percent_file, "w") as f2:
                    f2.write(percent_data.to_csv())


convert_data_to_percents()

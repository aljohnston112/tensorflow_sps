import glob
import os

import pandas

from src.config import yahoo_data_folder, yahoo_data_percent_folder


def get_percent(row, open_col, close, high):
    if row['Open'] != 0:
        return (row[close] - row[open_col]) / row[open_col]
    else:
        return (row[close] - row[high]) / row[high]


def convert_data_to_percents(data_folder, date_col_name, open_col, close, high):
    file_names = [f for f in glob.glob(data_folder + "*.csv")]
    out_folder = yahoo_data_percent_folder
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    for data_file_name in file_names:
        with open(data_file_name, 'r') as f:
            percent_file = out_folder + os.path.basename(f.name)
            if not os.path.exists(percent_file):
                raw_data = pandas.read_csv(f)
                raw_data.pop(date_col_name)
                percent_profit = raw_data.apply(lambda row: get_percent(row, open_col, close, high), axis=1)
                percent_profit.drop(percent_profit.index[:1], inplace=True)
                percent_data = raw_data.pct_change()
                percent_data["Profit"] = percent_profit
                percent_data.drop(percent_data.index[:1], inplace=True)
                with open(percent_file, "w") as f2:
                    f2.write(percent_data.to_csv())


def convert_yahoo_data_to_percents():
    convert_data_to_percents(yahoo_data_folder, "Date", "Open", "Close", "High")

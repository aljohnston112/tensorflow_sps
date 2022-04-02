import glob
import os

import pandas

from src.config import yahoo_data_folder, yahoo_data_numbers_folder
from src.data_transformers.data_to_percents import get_percent


def get_year(row):
    year = row.split("-")[0]
    return int(year)


def get_month(row):
    month = row.split("-")[1]
    return int(month)


def convert_data_to_numbers():
    files = [f for f in glob.glob(yahoo_data_folder + "*.csv")]
    out_folder = yahoo_data_numbers_folder
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    for data_file in files:
        with open(data_file, 'r') as f:
            numbers_file = out_folder + os.path.basename(f.name)
            if not os.path.exists(numbers_file):
                raw_data = pandas.read_csv(f)
                date_data = raw_data.pop("Date")
                year = date_data.apply(lambda row: get_year(row))
                month = date_data.apply(lambda row: get_month(row))
                percent_profit = raw_data.apply(lambda row: get_percent(row), axis=1)
                percent_profit.drop(percent_profit.index[:1], inplace=True)
                percent_profit.reset_index(inplace=True, drop=True)
                raw_data["Year"] = year
                raw_data["Month"] = month
                raw_data.drop(raw_data.tail(1).index, inplace=True)
                raw_data.reset_index(inplace=True, drop=True)
                raw_data["Profit"] = percent_profit

                with open(numbers_file, "w") as f2:
                    f2.write(raw_data.to_csv())


convert_data_to_numbers()

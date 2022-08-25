import json
import os
from math import ceil
from os.path import isfile, join

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

IEX_symbol_file = "IEX_symbols"
alpaca_data_folder = "alpaca_data"


def get_iex_symbols():
    with open(IEX_symbol_file, "r") as f:
        symbols = json.loads(f.read())
    files = [f.split(".")[0] for f in os.listdir(alpaca_data_folder) if isfile(join(alpaca_data_folder, f))]
    symbols = [s for s in symbols if s not in files and "+" not in s and "=" not in s and "^" not in s and "#" not in s]
    return symbols


def download_all_data():
    symbols = get_iex_symbols()
    downloads_per_minute = 200
    symbols_per_download = ceil(len(symbols) / downloads_per_minute)
    for i in range(0, downloads_per_minute):
        download_data(symbols[i * symbols_per_download: (i + 1) * symbols_per_download])


def save_bars(bars_df):
    if not os.path.exists(alpaca_data_folder):
        os.mkdir(alpaca_data_folder)
    for stock_symbol in bars_df.index.levels[0]:
        csv_file = alpaca_data_folder + "/" + stock_symbol + ".csv"
        bars_df.loc[stock_symbol].to_csv(csv_file)


def download_data(symbols):
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    stock_client = StockHistoricalDataClient(api_key, secret_key)
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start="1792-01-01"
    )
    bars = stock_client.get_stock_bars(request_params)
    save_bars(bars.df)


if __name__ == "__main__":
    download_all_data()

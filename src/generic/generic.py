import operator
from enum import Enum
from functools import reduce

from src.config import yahoo_data_percent_folder
from src.data_transformers import files_prepper
from src.data_transformers.files_prepper import get_data_from_csv_path


class Duration(Enum):
    AVERAGE = 1
    TRANSIENT = 0


class Direction(Enum):
    UP = 1
    DOWN = 0


class AverageTracker:

    def __init__(self):
        self.last = None
        self.average = None
        self.i = 0

    def add_next(self, value):
        if self.average is None:
            self.average = 0
        self.average = (value + (self.i * self.average)) / (self.i + 1)
        self.i += 1
        self.last = value


def test(training_filepaths, validation_filepaths):
    label_key = "Profit"
    money = 1.0
    split = 10
    total_money = 0

    number_of_months = 12 * 10

    all_features = {}
    for filepath in training_filepaths:
        with open(filepath, "r") as f:
            features, labels = get_data_from_csv_path(filepath, label_key)
            all_features[filepath] = (features, labels)

    delete = [k for k, v in all_features.items() if v[0].shape[0] < number_of_months]
    for d in delete:
        del all_features[d]

    sliced_features = {}
    for k, v in all_features.items():
        sliced_features[k] = (v[0].iloc[-number_of_months:], v[1].iloc[-number_of_months:])

    print(sliced_features)
    average_trackers = {}
    for k, v in sliced_features.items():
        average_trackers[k] = AverageTracker()
    for i in range(0, int(number_of_months / 2)):
        for k, v in sliced_features.items():
            average_trackers[k].add_next(v[0]["Open"].iloc[i])

    to_buy = []
    bought = {}
    for i in range(int(number_of_months / 2), number_of_months):
        if len(bought) != 100:
            for k, v in sliced_features.items():
                average_trackers[k].add_next(v[0]["Open"].iloc[i])
                if average_trackers[k].average > 0 and v[0]["Open"].iloc[i] < 0:
                    to_buy.append(k)
            to_buy = [tb for tb in to_buy if tb not in bought.keys()]
            to_buy_average_open = (sorted(to_buy, key=lambda item: -average_trackers[item].average))
            to_buy_open = (sorted(to_buy, key=lambda item: sliced_features[item][0]["Open"].iloc[i]))
            to_buy_volume = (sorted(to_buy, key=lambda item: -sliced_features[item][0]["Volume"].iloc[i]))
            to_buy_points = (
                sorted(to_buy,
                       key=lambda item: to_buy_average_open.index(item) +
                                        to_buy_open.index(item) +
                                        to_buy_volume.index(item)
                       )
            )[:100]

        delete = []
        for b in bought.keys():
            r = reduce(operator.mul, [b+1 for b in bought[b]], 1)
            if r > 1:
                total_money += (money/split) * r
                delete.append(b)
            else:
                bought[b].append(sliced_features[b][1].iloc[i])
        for d in delete:
            del bought[d]

        n_to_buy = split - len(bought)
        print(f"To buy: {n_to_buy}")
        n = 0
        for tb in to_buy_points:
            if n == n_to_buy:
                break
            bought[tb] = [sliced_features[tb][1].iloc[i]]
            n += 1
        print(f"Month: {i}")
        print(f"Total money: {total_money}")

    for b, v in bought.items():
        r = reduce(operator.mul, [b + 1 for b in bought[b]], 1)
        total_money += (money / split) * r
    print(total_money)


def train():
    training_filepaths, testing_filepaths, validation_filepaths = files_prepper.random_split(yahoo_data_percent_folder)
    test(training_filepaths, validation_filepaths)


if __name__ == "__main__":
    train()

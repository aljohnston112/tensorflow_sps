import operator
from collections import defaultdict
from functools import reduce


class ProfitCalculator:

    def __init__(self, initial_investment, number_of_stocks):
        self.number_of_stocks = number_of_stocks
        self.money_per_stock = initial_investment / number_of_stocks
        self.cached_percents = defaultdict(list)

    def calculate_percents(self, percents):
        profit = 0

        remove = [name for name in self.cached_percents if name not in percents]
        for name in remove:
            profit += reduce(operator.mul, self.cached_percents[name], 1)
            self.cached_percents.pop(name)

        append = [name for name in self.cached_percents if name in percents]
        for name in append:
            self.cached_percents[name].append(percents[name])

        to_buy = self.number_of_stocks - len(self.cached_percents)
        items = list(percents.items())
        for i in range(0, to_buy):
            self.cached_percents[items[i][0]].append(items[i][1])

        return profit

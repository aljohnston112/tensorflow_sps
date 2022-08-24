from collections import defaultdict

from src.generic.profit_calculator import ProfitCalculator


def test():
    profit_calculator = ProfitCalculator(initial_investment=1.0, number_of_stocks=2)
    percents = defaultdict(lambda: 0.0)
    percents["a"] = .5
    percents["b"] = .25
    assert profit_calculator.calculate_percents(percents) == 0
    percents["c"] = .5
    percents["d"] = .25
    assert profit_calculator.calculate_percents(percents) == 0
    percents.pop("a")
    assert profit_calculator.calculate_percents(percents) == .5*.5


if __name__ == "__main__":
    test()
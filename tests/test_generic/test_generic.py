import pandas

from src.generic.generic import AverageTracker, Direction, Duration


def test_average_tracker():
    average_tracker = AverageTracker()
    assert average_tracker.average is None
    assert average_tracker.last is None
    assert average_tracker.i == 0
    average_tracker.add_next(0.5)
    assert average_tracker.average == 0.5
    assert average_tracker.last == 0.5
    average_tracker.add_next(1)
    assert average_tracker.average == 0.75
    assert average_tracker.last == 1


def test_what_to_buy():
    data: dict[str, pandas.DataFrame]
    averages_to_track: dict[str, Direction]
    transients_to_track: dict[str, Direction]
    importance: (str, tuple[Duration, Direction])
    data_index: int


if __name__ == "__main__":
    test_average_tracker()

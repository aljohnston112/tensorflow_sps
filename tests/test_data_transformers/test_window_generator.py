from io import StringIO

import pandas as pd

from src.data_transformers.window_generator import WindowGenerator


def test():
    feature_width = 5
    label_width = 1
    label_offset = 0
    delta_i = 2
    features = pd.read_csv(StringIO("A\n1\n2\n3\n4\n5\n6\n7\n8"))
    labels = pd.read_csv(StringIO("A\n11\n12\n13\n14\n15\n16\n17\n18"))

    wg = WindowGenerator(
        feature_width=feature_width,
        label_width=label_width,
        label_offset=label_offset,
        delta_i=delta_i,
        features=features,
        labels=labels
    )
    for _ in range(7):
        o, g = wg.split_window()
        print(f"Features:\n{o}")
        print(f"Labels:\n{g}")

test()
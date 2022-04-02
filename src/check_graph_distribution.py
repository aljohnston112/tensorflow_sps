import itertools

import pandas as pd

from src.config import yahoo_data_dataset_folder
from src.data_transformers.files_prepper import __get_data_filepaths__
from src.data_transformers.window_generator import WindowGenerator

domain = [tuple(i) for i in list(itertools.product([0, 1], repeat=14))]
domain_map = {k: 0 for k in domain}
for filepath in __get_data_filepaths__(yahoo_data_dataset_folder):
    feature_width = 2
    delta_i = 1
    single_features = pd.read_csv(filepath, index_col=[0])
    wg = WindowGenerator(
        feature_width=feature_width,
        delta_i=delta_i,
        features=single_features
    )
    window_features = wg.split_window()
    while window_features is not None:
        window_features = window_features.to_numpy().flatten()
        window_features = pd.DataFrame([window_features], columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        domain_occurrences = window_features.groupby(window_features.columns.tolist(), as_index=False).size()
        count = domain_occurrences.pop("size")
        domain_occurrences = dict(zip([tuple(i) for i in domain_occurrences.values.tolist()], count.to_list()))
        for domain_input, size in domain_occurrences.items():
            domain_map[domain_input] += size
        window_features = wg.split_window()

total = 0
for t in domain_map.values():
    total += t
for domain_input, size in domain_map.items():
    domain_map[domain_input] = size/total
print()

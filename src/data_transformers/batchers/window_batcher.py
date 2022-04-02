import tensorflow as tf

from src.data_transformers.batchers.base_batcher import BaseBatcher
from src.data_transformers.files_prepper import get_data_from_csv_path
from src.data_transformers.normalizer import normalize_tensors
from src.data_transformers.window_generator import WindowGenerator


class WindowBatcher(BaseBatcher):
    def reset(self):
        self.i = 0

    def __init__(self, csv_filepaths, label_key):
        self.label_key = label_key
        self.filepaths = csv_filepaths
        self.i = 0
        for filepath in self.filepaths:
            single_features, single_labels = get_data_from_csv_path(filepath, label_key)
            feature_width = 10
            label_width = 1
            label_offset = 0
            delta_i = 1
            wg = WindowGenerator(
                feature_width=feature_width,
                label_width=label_width,
                label_offset=label_offset,
                delta_i=delta_i,
                features=single_features,
                labels=single_labels
            )
            window_features, window_labels = wg.split_window()
            self.time_features = []
            self.time_labels = []
            while window_features is not None and window_labels is not None and len(window_features) > 0 and len(window_labels) > 0:
                self.time_features.append(window_features)
                self.time_labels.append(window_labels)
                window_features, window_labels = wg.split_window()

    def get_single_data_batch(self):
        if self.i >= len(self.time_features):
            return None, None
        features = self.time_features[self.i]
        features = normalize_tensors(features)
        labels = self.time_labels[self.i]
        features = tf.reshape(features, [-1])
        features = tf.expand_dims(features, 0)
        self.i += 1
        return features, labels

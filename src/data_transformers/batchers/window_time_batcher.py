import tensorflow as tf

from src.data_transformers.batchers.base_batcher import BaseBatcher
from src.data_transformers.files_prepper import get_data_from_csv_path
from src.data_transformers.window_generator import WindowGenerator


class WindowTimeBatcher(BaseBatcher):
    def __init__(self, csv_filepaths, label_key):
        self.label_key = label_key
        self.filepaths = csv_filepaths
        self.i = 0
        time_features = []
        time_labels = []
        self.time_features = []
        self.time_labels = []
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
            features = []
            labels = []
            while window_features is not None and window_labels is not None:
                features.append(window_features)
                labels.append(window_labels)
                window_features, window_labels = wg.split_window()
            if len(features) > 0:
                time_features = tf.concat(features, 0)
                time_labels = tf.concat(labels, 0)
                s = [len(labels), feature_width]
                for i in time_features[0].shape[:]:
                    s.append(i)
                time_features = tf.reshape(time_features, s)
                time_features = tf.cast(time_features, tf.float32)
        self.time_features.append(time_features)
        self.time_labels.append(time_labels)

    def get_single_data_batch(self):
        if self.i >= len(self.time_features):
            return None, None
        features = self.time_features[self.i]
        labels = self.time_labels[self.i]
        self.i += 1
        return features, labels

    def reset(self):
        self.i = 0

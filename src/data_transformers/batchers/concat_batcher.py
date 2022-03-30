import tensorflow as tf

from src.data_transformers.batchers.base_batcher import BaseBatcher
from src.data_transformers.files_prepper import get_data_from_csv_path


class ConcatBatcher(BaseBatcher):

    def reset(self):
        self.i = 0

    def __init__(self, csv_filepaths, label_key, batch_size=0):
        self.label_key = label_key
        self.filepaths = csv_filepaths
        self.i = 0
        self.batch_size = batch_size
        features = []
        labels = []
        for filepath in self.filepaths:
            single_features, single_labels = get_data_from_csv_path(filepath, label_key)
            tf.debugging.check_numerics(single_features, f"{filepath} has NAN")
            tf.debugging.check_numerics(single_labels, f"{filepath} has NAN")
            features.append(single_features)
            labels.append(single_labels)
        self.features, self.labels = tf.concat(features, 0), tf.concat(labels, 0)
        if self.batch_size == 0:
            self.batch_size = len(self.features) - 1

    def get_single_data_batch(self):
        if self.i + self.batch_size >= len(self.features):
            return None, None

        if self.i + self.batch_size < len(self.features):
            features = self.features[self.i: self.i + self.batch_size]
            labels = self.labels[self.i: self.i + self.batch_size]
        else:
            features = self.features[self.i:]
            labels = self.labels[self.i:]
        self.i += self.batch_size
        return features, labels

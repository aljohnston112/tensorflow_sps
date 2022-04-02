

class WindowGenerator:
    def __init__(self, feature_width, delta_i, features,
                 label_width=0, label_offset=0, labels=None
                 ):

        self.features = features
        self.labels = labels

        self.feature_width = feature_width
        self.label_width = label_width
        self.label_offset = label_offset
        self.i = 0
        self.delta_i = delta_i

    def split_window(self):
        end_i = self.i + self.feature_width - 1 + self.label_width + self.label_offset
        if self.i + self.feature_width > len(self.features):
            if self.labels is not None:
                return None, None
            else:
                return None
        if self.labels is not None:
            if end_i > len(self.labels):
                return None, None

        if self.i + self.feature_width < len(self.features):
            features = self.features[self.i: self.i + self.feature_width]
        else:
            features = self.features[self.i:]

        start_i = self.i + self.feature_width - 1 + self.label_offset
        if self.labels is not None:
            if end_i < len(self.labels):
                labels = self.labels[start_i: end_i]
            else:
                labels = self.labels[start_i:]
            self.i += self.delta_i
            return features, labels

        self.i += self.delta_i
        return features

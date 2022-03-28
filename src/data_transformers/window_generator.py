

class WindowGenerator:
    def __init__(self, feature_width, label_width, label_offset, delta_i,
                 features, labels
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
        if self.i + self.feature_width > len(self.features) or end_i > len(self.labels):
            return None, None

        if self.i + self.feature_width < len(self.features):
            features = self.features[self.i: self.i + self.feature_width]
        else:
            features = self.features[self.i:]

        start_i = self.i + self.feature_width - 1 + self.label_offset
        if end_i < len(self.labels):
            labels = self.labels[start_i: end_i]
        else:
            labels = self.labels[start_i:]

        self.i += self.delta_i
        return features, labels

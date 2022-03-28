from tensorflow import keras


def normalize_tensors(features):
    layer = keras.layers.Normalization()
    layer.adapt(features)
    return layer(features)

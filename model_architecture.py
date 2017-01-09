from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

def crop(x):
    import tensorflow as tf
    return x[:, 60:134, 0:320]

def normalize(x):
    return x / 127.5 - 1

def model_architecture():
    model = Sequential()
    model.add(Lambda(crop, input_shape=(160, 320, 3), name="crop"))
    model.add(Lambda(normalize, name="normalize"))

    model.add(Convolution2D(32, 2, 2, subsample=(2, 2), border_mode="same"))
    model.add(LeakyReLU())

    model.add(Convolution2D(64, 2, 2, subsample=(2, 2), border_mode="same"))
    model.add(LeakyReLU())

    model.add(MaxPooling2D())

    model.add(Convolution2D(2, 2, 2, subsample=(2, 2), border_mode="same"))
    model.add(LeakyReLU())

    model.add(Convolution2D(4, 2, 2, subsample=(2, 2), border_mode="same"))
    model.add(LeakyReLU())

    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(100))
    model.add(LeakyReLU())

    model.add(Dense(50))
    model.add(LeakyReLU())

    model.add(Dense(1))

    return model

if __name__ == '__main__':
    model = model_architecture()
    model.summary()

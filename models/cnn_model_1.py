from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

def crop(x):
    import tensorflow as tf
    return x[:, 60:134, 0:320]

def resize(x):
    import tensorflow as tf
    return tf.image.resize_images(x, (37, 160))

def normalize(x):
    return x / 127.5 - 1

def cnn_model_1():
    model = Sequential()
    model.add(Lambda(crop, input_shape=(160, 320, 3), name="crop"))
    # model.add(Lambda(resize, name="resize"))
    model.add(Lambda(normalize, name="normalize"))

    model.add(Convolution2D(10, 2, 2, subsample=(2, 2), border_mode="valid"))
    model.add(LeakyReLU())

    model.add(Convolution2D(10, 2, 2, subsample=(2, 2), border_mode="valid"))
    model.add(LeakyReLU())

    model.add(MaxPooling2D())

    model.add(Convolution2D(10, 2, 2, subsample=(2, 2), border_mode="valid"))
    model.add(LeakyReLU())

    model.add(Convolution2D(10, 2, 2, subsample=(2, 2), border_mode="valid"))
    model.add(LeakyReLU())

    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(100))
    model.add(LeakyReLU())

    model.add(Dense(50))
    model.add(LeakyReLU())

    model.add(Dense(1))

    return model

if __name__ == "__main__":
    print(cnn_model_1().summary())

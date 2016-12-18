from generate import generate
from save import save
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.applications.inception_v3 import InceptionV3
import numpy as np
import json
import math

nb_epochs = 12
nb_proportion = 4
total_examples = 14424
pct_train = 0.8
pct_valid = 0.1
pct_test = 0.1

nb_train = math.floor(math.floor(total_examples * pct_train) / nb_proportion)
nb_train = math.floor(math.floor(total_examples * pct_train) / nb_proportion)
nb_valid = math.floor(math.floor(total_examples * pct_valid) / nb_proportion)
nb_test = math.floor(total_examples * pct_test)

train, valid, test = generate("data/driving_log.csv",
                              pct_train=pct_train,
                              pct_valid=pct_valid,
                              pct_test=pct_test)

def resize(x):
    import tensorflow as tf
    return tf.image.resize_images(x, (80, 160))

def normalize(x):
    return x / 127.5 - 1

model = Sequential()
model.add(Lambda(resize, input_shape=(160, 320, 3), name="resize"))
model.add(Lambda(normalize, name="normalize"))
model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(1))
model.add(LeakyReLU())

model.compile(loss='mse', optimizer="adam")

history = model.fit_generator(generator=train,
                              samples_per_epoch=nb_train,
                              nb_epoch=nb_epochs,
                              validation_data=valid,
                              nb_val_samples=nb_valid,
                              callbacks=[])


h = history.history
print("training loss: {}".format(h["loss"][-1]))
print("validation loss: {}".format(h["val_loss"][-1]))

out = model.evaluate_generator(test, val_samples=nb_test)
print("test loss: {}".format(out))

save(model)

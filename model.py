from generate import generate
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
import numpy as np
import json

samples_per_epoch=128
nb_epochs = 1

train, valid, test = generate("data/driving_log.csv",
                              pct_train=0.8,
                              pct_valid=0.1,
                              pct_test=0.1)

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

model.compile(loss='mse', optimizer=Adam(lr=0.1))

history = model.fit_generator(generator=train,
                              samples_per_epoch=samples_per_epoch,
                              nb_epoch=nb_epochs,
                              validation_data=valid,
                              nb_val_samples=256,
                              callbacks=[])


h = history.history
print("training loss: {}".format(h["loss"][-1]))
print("validation loss: {}".format(h["val_loss"][-1]))

j = model.to_json()
with open("model.json", "w") as f:
    json.dump(j, f)

model.save_weights("model.h5")

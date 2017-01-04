from generate import generate
from keras.callbacks import ModelCheckpoint
from models.cnn_model_1 import cnn_model_1
import numpy as np
import json
import math

def save_model_architecture(model, filename="model.json"):
    with open(filename, "w") as f:
        json.dump(model.to_json(), f)

def checkpoint(filepattern="model.{epoch:02d}.h5"):
    return ModelCheckpoint(filepattern, save_weights_only=True)

nb_epochs = 100
nb_proportion = 1
total_examples = 24108
pct_train = 0.9
pct_valid = 0.1
pct_test = 0.0

nb_train = math.floor(math.floor(total_examples * pct_train) / nb_proportion)
nb_train = math.floor(math.floor(total_examples * pct_train) / nb_proportion)
nb_valid = math.floor(math.floor(total_examples * pct_valid) / nb_proportion)
nb_test = math.floor(total_examples * pct_test)

train, valid, test = generate("data/driving_log.csv",
                              pct_train=pct_train,
                              pct_valid=pct_valid,
                              pct_test=pct_test)

def crop(x):
    import tensorflow as tf
    return x[:, 60:134, 0:320]

def normalize(x):
    return x / 127.5 - 1

model = Sequential()
model.add(Lambda(crop, input_shape=(160, 320, 3), name="crop"))
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

model.compile(loss='mse', optimizer=Adam(lr=0.0001))
save_model_architecture(model)
model.fit_generator(generator=train,
                    samples_per_epoch=nb_train,
                    nb_epoch=nb_epochs,
                    nb_val_samples=nb_valid,
                    validation_data=valid,
                    callbacks=[checkpoint()])

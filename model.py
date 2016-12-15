from preprocess import preprocess
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential
import numpy as np

epochs = 1
size = (16,32, 3)
flatten = False

train_data = preprocess(datadir="data", filename="train_driving_log.csv", size=size, flatten=flatten)
valid_data = preprocess(datadir="data", filename="valid_driving_log.csv", size=size, flatten=flatten)
test_data = preprocess(datadir="data", filename="test_driving_log.csv", size=size, flatten=flatten)

model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=size))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid'))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid'))
model.add(Flatten())
# model.add(Dense(1, input_dim=np.prod(size)))
model.add(Dense(1))
model.add(Activation('relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit_generator(generator=train_data,
                              samples_per_epoch=128,
                              nb_epoch=epochs,
                              validation_data=valid_data,
                              nb_val_samples=1000)

h = history.history
print("training accuracy: {}".format(h["acc"][-1]))
print("validation accuracy: {}".format(h["val_acc"][-1]))

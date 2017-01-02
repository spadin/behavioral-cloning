from generate import generate
from save import save, save_json
from keras.callbacks import ModelCheckpoint
from models.cnn_model_1 import cnn_model_1
import numpy as np
import json
import math

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

model = cnn_model_1()

checkpoint = ModelCheckpoint(filepath="model.{epoch:02d}.h5", save_weights_only=True)
save_json(model)
history = model.fit_generator(generator=train,
                              samples_per_epoch=nb_train,
                              nb_epoch=nb_epochs,
                              nb_val_samples=nb_valid,
                              validation_data=valid,
                              callbacks=[checkpoint])


h = history.history
print("training loss: {}".format(h["loss"][-1]))
# print("validation loss: {}".format(h["val_loss"][-1]))
# out = model.evaluate_generator(test, val_samples=nb_test)
# print("test loss: {}".format(out))
# save(model)

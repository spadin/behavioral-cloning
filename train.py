from generate import generate
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import json
import math
import sys
import json

def load_model(filename):
    json_file = sys.argv[1]
    with open(json_file, 'r') as f:
        j = json.load(f)
        return model_from_json(j)

def checkpoint(filepattern="model.retrain.{epoch:02d}.h5"):
    return ModelCheckpoint(filepattern, save_weights_only=True)

nb_epochs = 10
nb_proportion = 1
total_examples = 24108
pct_train = 1.0
pct_valid = 0.0
pct_test = 0.0

nb_train = math.floor(math.floor(total_examples * pct_train) / nb_proportion)
nb_train = math.floor(math.floor(total_examples * pct_train) / nb_proportion)
nb_valid = math.floor(math.floor(total_examples * pct_valid) / nb_proportion)
nb_test = math.floor(total_examples * pct_test)

train, valid, test = generate("data/driving_log.csv",
                              pct_train=pct_train,
                              pct_valid=pct_valid,
                              pct_test=pct_test)

model = load_model(sys.argv[1])
model.load_weights(json_file.replace("json", "h5"))
model.compile(loss='mse', optimizer=Adam(lr=0.0001))

model.fit_generator(generator=train,
                    samples_per_epoch=nb_train,
                    nb_epoch=nb_epochs,
                    callbacks=[checkpoint()])

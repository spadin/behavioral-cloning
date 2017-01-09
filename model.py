from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from model_architecture import model_architecture
from generator import Generator
import numpy as np
import json

def save_model_architecture(model, filename="model.json"):
    with open(filename, "w") as f:
        json.dump(model.to_json(), f)

def checkpoint(filepattern="model.{epoch:02d}.h5"):
    return ModelCheckpoint(filepattern, save_weights_only=True)

nb_epochs = 100
pct_train = 0.9
pct_valid = 0.1
pct_test = 0.0

generator = Generator("data/driving_log.csv",
                      pct_train,
                      pct_valid,
                      pct_test)

total_examples = generator.examples()

model = model_architecture()
model.compile(loss='mse', optimizer=Adam(lr=0.0001))
save_model_architecture(model)
model.fit_generator(generator=generator.train(),
                    samples_per_epoch=generator.nb_train_examples(),
                    nb_epoch=nb_epochs,
                    nb_val_samples=generator.nb_valid_examples(),
                    validation_data=generator.valid(),
                    callbacks=[checkpoint()])

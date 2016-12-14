WIP, this doesn't work yet.

# Behavioral Cloning Project

## Before you begin

You'll need some data. You can generate your data by running the simulator in
training mode, or download the [dataset from class.][1]

## Splitting the data

The data from class needs to be split into a training set, validation set, and
test set. This can be done any number of times, but must be done at least once.
This will write three files that can be loaded later for extracting features
from related images.

```sh
$ python split.py
```

Writes:

  * data/train_driving_log.csv
  * data/valid_driving_log.csv
  * data/test_driving_log.csv

## Training the model

```sh
$ python model.py
```

Training the model will result in two files being generated. `model.json`
includes the model architecture. `model.h5` includes the model weights that
were just trained.

Note: if the model doesn't find training, validation, and test datasets in
`data` it will try running `prepare`.

## Run the drive server

Once you've trained the model, or if you have a `model.json` and `model.h5`
files, you can run the driving server.

```sh
$ python drive.py model.json
```

[1]: https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

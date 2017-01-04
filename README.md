# Behavioral Cloning Project

## Table of Contents

1. [About the model](#about-the-model)
1. [Training dataset](#training-dataset)
1. [Before you begin](#before-you-begin)
1. [Splitting the data](#splitting-the-data)
1. [Training the model](#training-the-model)
1. [Running the drive server](#running-the-drive-server)

## About the model

A model with the following architecture is used.

![Model architecture](./model.png?raw=true)

## Training dataset

The model was trained on an AWS g2.2xlarge EC2 instance. The model ran for a
total of 100 epochs. At each epoch, the checkpoint of the model would be
created, which saved the weights at that epoch.

I ran the model after each epoch in the simulator. The first successful lap
was done at epoch 17. But, no other epoch did a successful lap until epoch 34.

The left, right and center images are used in training. I've added 0.1 (2.5
degrees) to the left images and -0.1 to the right images to compensate for the
difference in camera angle.

Here's an example of the three different types of images.

**Left image**

![Left image](./left_2016_12_19_20_10_35_002.jpg?raw=true)

**Center image**

![Center image](./center_2016_12_19_20_10_35_002.jpg?raw=true)

**Right image**

![Right image](./right_2016_12_19_20_10_35_002.jpg?raw=true)

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

## Running the drive server

Once you've trained the model, or if you have a `model.json` and `model.h5`
files, you can run the driving server.

```sh
$ python drive.py model.json
```

[1]: https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

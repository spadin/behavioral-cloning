from scipy.misc import imread
from shuffle import shuffle
from split import split
import csv
import numpy as np
import sys

def prepare_feature(image):
    return np.array([imread("{}/{}".format("data", image))]).astype(np.float32)

def prepare_label(steering):
    return np.array([float(steering)])

def read(filepath):
    with open(filepath) as csv_file:
        csvreader = csv.reader(csv_file, skipinitialspace=True)
        next(csvreader) # skip header row
        return [(row[0], row[1], row[2], row[3]) for row in csvreader]

def generator(data):
    shuffle(data)
    for center, left, right, steering in data:
        yield (prepare_feature(center), prepare_label(steering))
        yield (prepare_feature(left), prepare_label(float(steering) + 0.1))
        yield (prepare_feature(right), prepare_label(float(steering) - 0.1))

def infinite_generator(data):
    g = generator(data)
    while True:
        try:
            yield next(g)
        except StopIteration:
            g = generator(data)

def generate(filepath, pct_train, pct_valid, pct_test):
    """
    Create generators for the driving log data at filepath. Shuffles data the
    data before splitting and reshuffles again after each generator is
    exhausted.

    Note: pct_train + pct_valid + pct_test (must =) 1

    Arguments:
    filepath  -- path to driving_log.csv file
    pct_train -- (eg. 0.9) percentage split for training data
    pct_valid -- (eg. 0.1) percentage split for validation data
    pct_test  -- (eg. 0.0) percentage split for test data

    """
    data = read(filepath)
    shuffle(data)

    train, valid, test = split(data, pct_train, pct_valid, pct_test)

    return (infinite_generator(train),
            infinite_generator(valid),
            infinite_generator(test))

if __name__ == "__main__":
    generate("data/driving_log.csv", 0.9, 0.1, 0.0)

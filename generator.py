from scipy.misc import imread
from split import split
from random import shuffle
import csv
import math
import numpy as np

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
    for center, left, right, steering in data:
        yield (prepare_feature(center), prepare_label(steering))
        yield (prepare_feature(left), prepare_label(float(steering) + 0.1))
        yield (prepare_feature(right), prepare_label(float(steering) - 0.1))

class Generator:
    def __init__(self, filepath, pct_train, pct_valid, pct_test):
        self.filepath = filepath
        self.pct_train = pct_train
        self.pct_valid = pct_valid
        self.pct_test = pct_test
        self.data = read(filepath)
        self.resample()

    def examples(self):
        return len(self.data) * 3

    def nb_train_examples(self):
        return math.floor(self.examples() * self.pct_train)

    def nb_valid_examples(self):
        return math.floor(self.examples() * self.pct_valid)

    def nb_test_examples(self):
        return math.floor(self.examples() * self.pct_test)

    def split_data(self):
        data = split(self.data,
                     self.pct_train,
                     self.pct_valid,
                     self.pct_test)

        (self.train_data,
         self.valid_data,
         self.test_data) = data

    def create_generators(self):
        self.train_generator = generator(self.train_data)
        self.valid_generator = generator(self.valid_data)
        self.test_generator = generator(self.test_data)

    def shuffle_data(self):
        shuffle(self.data)

    def resample(self):
        self.shuffle_data()
        self.split_data()
        self.create_generators()

    def train(self):
        while True:
            try:
                yield next(self.train_generator)
            except StopIteration:
                self.resample()
                yield next(self.train_generator)

    def valid(self):
        while True:
            try:
                yield next(self.valid_generator)
            except StopIteration:
                self.resample()
                yield next(self.valid_generator)

    def test():
        while True:
            try:
                yield next(self.test_generator)
            except StopIteration:
                self.resample()
                yield next(self.test_generator)

from read import read
from shuffle import shuffle
from load import load_feature, load_label
from split import split

def prepare(data):
    for image, steering in data:
        yield (load_feature(image), load_label(steering))

def generator(data):
    shuffle(data)
    for image, steering in data:
        yield (load_feature(image), load_label(steering))

def infinite_generator(data):
    g = generator(data)
    while True:
        try:
            yield next(g)
        except StopIteration:
            g = generator(data)

def generate(filepath, pct_train, pct_valid, pct_test):
    data = read(filepath)
    train, valid, test = split(data, pct_train, pct_valid, pct_test)

    return (infinite_generator(train),
            infinite_generator(valid),
            infinite_generator(test))

if __name__ == "__main__":
    generate("data/driving_log.csv", 0.9, 0.1, 0.0)

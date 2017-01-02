from read import read
from shuffle import shuffle
from prepare import prepare_feature, prepare_label
from split import split

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
    data = read(filepath)
    shuffle(data)

    train, valid, test = split(data, pct_train, pct_valid, pct_test)

    return (infinite_generator(train),
            infinite_generator(valid),
            infinite_generator(test))

if __name__ == "__main__":
    generate("data/driving_log.csv", 0.9, 0.1, 0.0)

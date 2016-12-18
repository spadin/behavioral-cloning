from read import read
from shuffle import shuffle
from prepare import prepare_feature, prepare_label
from split import split

def generator(data):
    shuffle(data)
    for image, steering in data:
        yield (prepare_feature(image), prepare_label(steering))

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

    print("Generating data")
    print("total data length: {}".format(len(train) + len(valid) + len(test)))
    print("train length: {}".format(len(train)))
    print("valid length: {}".format(len(valid)))
    print("test length: {}".format(len(test)))

    return (infinite_generator(train),
            infinite_generator(valid),
            infinite_generator(test))

if __name__ == "__main__":
    generate("data/driving_log.csv", 0.9, 0.1, 0.0)

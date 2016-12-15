from driving_log.extract_features import ExtractFeatures
from scipy.misc import imresize
import numpy as np

def normalize_feature(feature, size, flatten):
    feature = np.array(feature)
    feature = imresize(feature, size)
    feature = feature.astype(np.float32)
    feature /= 255.
    feature -= 0.5

    if flatten == True:
        feature = feature.reshape(-1, np.prod(np.shape(feature)))
    else:
        feature = feature.reshape(-1, *np.shape(feature))

    return feature

def preprocess(datadir, filename, size, flatten):
    while True:
        datas = ExtractFeatures(datadir, filename).execute()

        for feature, label in datas:
            yield (normalize_feature(feature, size, flatten), np.array([label]))

if __name__ == "__main__":
    size = (16, 32)
    flatten = False
    train = preprocess(datadir="data", filename="train_driving_log.csv", size=size, flatten=flatten)
    valid = preprocess(datadir="data", filename="valid_driving_log.csv", size=size, flatten=flatten)
    test = preprocess(datadir="data", filename="test_driving_log.csv", size=size, flatten=flatten)

    first, _ = next(train)
    print(first.shape)

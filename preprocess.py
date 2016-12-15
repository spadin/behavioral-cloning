from driving_log.extract_features import ExtractFeatures
import numpy as np

def normalize_feature(feature):
    feature = np.array(feature)
    feature = feature.astype(np.float32)
    feature /= 255.
    feature -= 0.5
    return feature


def preprocess(datadir, filename):
    datas = ExtractFeatures(datadir, filename).execute()

    for feature, label in datas:
        yield (normalize_feature(feature), label)

if __name__ == "__main__":
    train = preprocess(datadir="data", filename="train_driving_log.csv")
    valid = preprocess(datadir="data", filename="valid_driving_log.csv")
    test = preprocess(datadir="data", filename="test_driving_log.csv")

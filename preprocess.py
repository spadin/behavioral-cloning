from driving_log.feature_extractor import FeatureExtractor
import numpy as np

class Preprocess:
    def __init__(self, filename, datadir="data"):
        self.extractor = FeatureExtractor(datadir, filename)

    def __normalize_feature(self, feature):
        feature = np.array(feature)
        feature = feature.astype(np.float32)
        feature /= 255.
        feature -= 0.5
        return feature

    def execute(self):
        for feature, label in self.extractor.features_and_labels():
            yield (self.__normalize_feature(feature), label)

if __name__ == "__main__":
    train = Preprocess(filename="train_driving_log.csv").execute()
    valid = Preprocess(filename="valid_driving_log.csv").execute()
    test = Preprocess(filename="test_driving_log.csv").execute()

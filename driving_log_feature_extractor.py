from driving_log_reader import DrivingLogReader
from scipy.misc import imread

class DrivingLogFeatureExtractor:
    def __init__(self, datadir="./data", filename="driving_log.csv"):
        self.datadir = datadir
        self.filename = filename
        self.__setup_data()

    def __setup_data(self):
        reader = DrivingLogReader(self.__datapath(self.filename))
        self.data = reader.data()

    def __datapath(self, image):
        return "{}/{}".format(self.datadir, image)

    def features(self):
        for image, steering in self.data:
            yield imread(self.__datapath(image))

    def labels(self):
        for image, steering in self.data:
            yield steering

if __name__ == "__main__":
    extractor = DrivingLogFeatureExtractor()

    i = 0
    features = extractor.features()
    while i < 1:
        feature = next(features)
        print(feature)
        i += 1

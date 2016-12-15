from .reader import Reader
from scipy.misc import imread

class ExtractFeatures:
    """
    Lazily extracts features from the driving log file using a DrivingLogReader
    to iterate through the driving log data
    """
    def __init__(self, datadir="./data", filename="driving_log.csv"):
        self.datadir = datadir
        self.filename = filename
        self.__setup_data()

    def __setup_data(self):
        reader = Reader(self.__datapath(self.filename))
        self.data = reader.data()

    def __datapath(self, image):
        return "{}/{}".format(self.datadir, image)

    def features(self):
        for image, steering in self.data:
            yield imread(self.__datapath(image))

    def labels(self):
        for image, steering in self.data:
            yield steering

    def execute(self):
        features = self.features()
        labels = self.labels()
        return zip(features, labels)

if __name__ == "__main__":
    extractor = ExtractFeatures()

    i = 0
    data = extractor.execute()
    while i < 1:
        feature, label = next(data)
        print(feature, label)
        i += 1

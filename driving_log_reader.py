import csv

class DrivingLogReader:
    """
    Helper class for loading and formatting driving log data.
    """
    def __init__(self, filepath):
        """
        Arguments:

        filepath - driving log filepath.
        """
        self.filepath = filepath

    def __format_row(self, row):
        """
        Formats a csv row and returns a list with only the fields we care
        about.
        """
        image = row[0]
        steering = row[3]

        return (image, steering)

    def __csv_rows(self):
        """
        Opens the driving log csv file and yields each row.
        """
        with open(self.filepath) as csv_file:
            csvreader = csv.reader(csv_file, skipinitialspace=True)
            next(csvreader) # skip header row
            for row in csvreader:
                yield row

    def data(self):
        """
        Formats each csv row and yields data formatted as a list with an
        image path and a steering angle.

        Note: data is yielded, not returned, so it can be iterated in a lazily.

        Example yield:
        [('IMG/center_2016_12_01_13_46_23_394.jpg', ' -0.02177976'),
         ('IMG/center_2016_12_01_13_46_23_497.jpg', ' -0.1262177')]
        """
        return (self.__format_row(row) for row in self.__csv_rows())

if __name__ == "__main__":
    reader = DrivingLogReader("./data/driving_log.csv")

    print("Formatted data from ./data/driving_log.csv")

    for data in reader.data():
        print(data)

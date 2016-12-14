import csv
import math

class SplitTrainValidTest:
    def __init__(self, outdir='data', infilename="driving_log.csv", pct_train=0.5, pct_valid=0.25, pct_test=0.25):
        self.infilename = infilename
        self.outdir = outdir
        self.pct_train = pct_train
        self.pct_valid = pct_valid
        self.pct_test = pct_test

    def __filepath(self, filename):
        return "{}/{}".format(self.outdir, filename)

    def __infilepath(self):
        return self.__filepath(self.infilename)

    def __train_filepath(self):
        return self.__filepath("train_driving_log.csv")

    def __valid_filepath(self):
        return self.__filepath("valid_driving_log.csv")

    def __test_filepath(self):
        return self.__filepath("test_driving_log.csv")

    def __read_infile(self):
        with open(self.__infilepath(), "r") as infile:
            csvreader = csv.reader(infile)
            header = next(csvreader)
            rows = []

            for row in csvreader:
                rows.append(row)

        return (header, rows)

    def __slices(self, rows):
        nb_rows = len(rows)

        split_1 = math.floor(nb_rows * self.pct_train)
        split_2 = split_1 + math.floor(nb_rows * self.pct_valid)
        split_3 = split_2 + math.floor(nb_rows * self.pct_test)

        train = slice(0, split_1)
        valid = slice(split_1, split_2)
        test = slice(split_2, split_3)

        return (rows[train], rows[valid], rows[test])

    def __write(self, outfilepath, header, rows):
        with open(outfilepath, "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(header)
            csvwriter.writerows(rows)

    def execute(self):
        header, rows = self.__read_infile()
        train_slice, valid_slice, test_slice = self.__slices(rows)

        self.__write(self.__train_filepath(), header, train_slice)
        self.__write(self.__valid_filepath(), header, valid_slice)
        self.__write(self.__test_filepath(), header, test_slice)

if __name__ == "__main__":
    split = SplitTrainValidTest()
    split.execute()

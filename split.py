import csv
import math

def filepath(dirname, filename):
    return "{}/{}".format(dirname, filename)

def infilepath(dirname, filename):
    return filepath(dirname, filename)

def outfilepaths(dirname):
    return (filepath(dirname, "train_driving_log.csv"),
            filepath(dirname, "valid_driving_log.csv"),
            filepath(dirname, "test_driving_log.csv"))

def read(filepath):
    with open(filepath, "r") as f:
        csvreader = csv.reader(f)
        header = next(csvreader)
        rows = [row for row in csvreader]

    return (header, rows)

def write(filepath, header, rows):
    with open(filepath, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(header)
        csvwriter.writerows(rows)

def slices(rows, pct_train, pct_valid, pct_test):
    nb_rows = len(rows)

    split_1 = math.floor(nb_rows * pct_train)
    split_2 = split_1 + math.floor(nb_rows * pct_valid)
    split_3 = split_2 + math.floor(nb_rows * pct_test)

    return (slice(0, split_1),
            slice(split_1, split_2),
            slice(split_2, split_3))

def slice_rows(rows, pct_train, pct_valid, pct_test):
    train, valid, test = slices(rows, pct_train, pct_valid, pct_test)
    return (rows[train], rows[valid], rows[test])

def split_train_valid_test(dirname='data', infilename="driving_log.csv", pct_train=0.5, pct_valid=0.25, pct_test=0.25):
    infile = infilepath(dirname, infilename)
    (header, rows) = read(infile)
    train_slice, valid_slice, test_slice = slice_rows(rows, pct_train, pct_valid, pct_test)
    train_filepath, valid_filepath, test_filepath = outfilepaths(dirname)

    write(train_filepath, header, train_slice)
    write(valid_filepath, header, valid_slice)
    write(test_filepath, header, test_slice)

if __name__ == "__main__":
    split_train_valid_test()

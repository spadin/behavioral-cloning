import math

def row_slices(rows, pct_train, pct_valid, pct_test):
    """
    Returns a three slices that can be used to split the row data into
    training, validation and test data. There is no overlap in the slices, so
    validation and test data will not be in the training data.
    """
    nb_rows = len(rows)

    split_1 = math.floor(nb_rows * pct_train)
    split_2 = split_1 + math.floor(nb_rows * pct_valid)
    split_3 = split_2 + math.floor(nb_rows * pct_test)

    return (slice(0, split_1),
            slice(split_1, split_2),
            slice(split_2, split_3))

def split(data, pct_train, pct_valid, pct_test):
    """
    Returns data split into three parts, each part representing training,
    validation and test data repectively.

    Note: pct_train + pct_valid + pct_test must equal 1. And each should be
    between 0.0 and 1.0.
    """
    train_slice, valid_slice, test_slice = row_slices(data, pct_train, pct_valid, pct_test)

    return (data[train_slice],
            data[valid_slice],
            data[test_slice])


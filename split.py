import math

def row_slices(rows, pct_train, pct_valid, pct_test):
    nb_rows = len(rows)

    split_1 = math.floor(nb_rows * pct_train)
    split_2 = split_1 + math.floor(nb_rows * pct_valid)
    split_3 = split_2 + math.floor(nb_rows * pct_test)

    return (slice(0, split_1),
            slice(split_1, split_2),
            slice(split_2, split_3))

def split(data, pct_train, pct_valid, pct_test):
    train_slice, valid_slice, test_slice = row_slices(rows, pct_train, pct_valid, pct_test)

    return (rows[train_slice],
            rows[valid_slice],
            rows[test_slice])


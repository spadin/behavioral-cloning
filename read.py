import csv
import sys

def read(filepath):
    with open(filepath) as csv_file:
        csvreader = csv.reader(csv_file, skipinitialspace=True)
        # next(csvreader) # skip header row
        return [(row[0], row[1], row[2], row[3]) for row in csvreader]

if __name__ == "__main__":
    driving_log = sys.argv[1]
    print(read(driving_log))

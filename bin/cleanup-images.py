# files=$(find ./data/IMG | cut -c 12-)
import os
import csv
import numpy as np

def actual_image_files():
    return set(np.array(os.listdir("data/IMG")))

def files_in_csv():
    with open("data/driving_log.csv") as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        files = [(row[0].replace("IMG/",""), row[1].replace("IMG/",""), row[2].replace("IMG/","")) for row in reader]
        files = np.array(files)
        return set(files.flatten())

if __name__ == "__main__":
    """
    Deletes image files that are not referenced in the driving_log.csv file
    """
    difference = actual_image_files() - files_in_csv()

    print("Deleting {} images".format(len(difference)))

    for filename in difference:
        os.remove("{}/{}".format("data/IMG", filename))

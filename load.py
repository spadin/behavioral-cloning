from scipy.misc import imread
import numpy as np

def load_feature(image):
    return np.array([imread("{}/{}".format("data", image))]).astype(np.float32)

def load_label(steering):
    return np.array([float(steering)])

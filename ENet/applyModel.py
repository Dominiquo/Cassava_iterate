import numpy as np


def output_to_mask(output, shape=(256,256), threshold=.5):
    yval = [1 if l[1] >= threshold else 0 for l in output]
    ymask = np.reshape(yval,(256,256))
    return ymask.astype(bool)
import sys
from pathlib import Path
from os.path import realpath
path = Path(__file__).parent.parent.absolute()
sys.path.append(realpath(path)) # Project root folder
from sklearn.metrics import pairwise_distances
import numpy as np

def sim_matrix(df):
    dist_matrix = pairwise_distances(df.values, metric='euclidean')
    gamma = 1/df.values.shape[1]
    gamma = 0.001
    return np.exp(-np.square(dist_matrix) * gamma)

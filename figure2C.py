# refactoring https://github.com/nicoring/hippocampus-predictive-map
# this code intended for replicatig the figure 2C.
# analytic compute SR matrix from Transitional matrix.

# load libray
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mazemaking as mm
import sr



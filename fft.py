import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum

class Dimension(Enum):
    X = 0
    Y = 1
    Z = 2

UPPER_WAVEGUIDE_FILE_PREFIX = "m.region7"
LOWER_WAVEGUIDE_FILE_PREFIX = ""
INPUT_DIR = './main.out'

# load data
m = np.load(os.path.join(INPUT_DIR, f'{UPPER_WAVEGUIDE_FILE_PREFIX}000000.npy'))
print(m.shape)
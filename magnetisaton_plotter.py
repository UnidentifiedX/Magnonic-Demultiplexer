import numpy as np
import matplotlib.pyplot as plt
import os
from mx3_utils import Dimension

FILE_PREFIX = "m"
INPUT_DIR = './template.out'
DT = 50e-12
FRAME_COUNT = 1001
DIMENSION = Dimension.X

x_start, x_end = 224, 225
y_start, y_end = 35, 50

# === LOAD DATA ===
data = np.load(os.path.join(INPUT_DIR, f'{FILE_PREFIX}001000.npy'))[Dimension.X.value, 0]

# === PLOT ===
plt.figure(figsize=(10, 4))
vabs = np.max(np.abs(data))
plt.imshow(data, cmap='seismic', vmin=-vabs, vmax=vabs, origin='lower')
plt.colorbar(label='Magnetisation (arb. units)')
plt.title(f'm[{["x", "y", "z"][Dimension.X.value]}]')

plt.xlabel('X index')
plt.ylabel('Y index')
plt.tight_layout()
plt.show()
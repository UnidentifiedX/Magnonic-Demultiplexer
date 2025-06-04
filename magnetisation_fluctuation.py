import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from enum import Enum

class Dimension(Enum):
    X = 0
    Y = 1
    Z = 2

# settings
INPUT_DIR = './main.out'
OUTPUT_VIDEO = 'fluctuation_video.mp4'
FRAME_COUNT = 82
FPS = 10
DIMENSION = Dimension.X.value
Z_SLICE = 5

ref_frame = np.load(os.path.join(INPUT_DIR, 'm000000.npy'))[DIMENSION, Z_SLICE]
height, width = ref_frame.shape

video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    FPS,
    (width, height)
)

# === Function: fluctuation to BGR image ===
def to_bgr_image(array2d):
    cmap = plt.get_cmap('seismic')
    norm = array2d / (np.max(np.abs(array2d)) + 1e-12)
    rgba = cmap(norm)
    rgb = np.delete(rgba, 3, 2)  # drop alpha
    bgr = (rgb[..., ::-1] * 255).astype(np.uint8)
    return np.ascontiguousarray(bgr)

# === Loop through frames and write video ===
for i in range(FRAME_COUNT):
    filepath = os.path.join(INPUT_DIR, f'm{i:06d}.npy')
    data_slice = np.load(filepath)[DIMENSION, Z_SLICE]
    fluct = data_slice - ref_frame
    frame_img = to_bgr_image(fluct)
    video_writer.write(frame_img)

video_writer.release()
print(f'Fluctuation video saved to: {OUTPUT_VIDEO}')
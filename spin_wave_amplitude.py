import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from enum import Enum

class Dimension(Enum):
    X = 0
    Y = 1
    Z = 2

# Settings
FRAME_COUNT = 22
INPUT_DIR = './main.out'
OUTPUT_VIDEO = 'spin_wave_cv2.mp4'
FPS = 10
DIMENSION = Dimension.X.value # adjust for different dimensions
Z_SLICE = 5

def get_normalised_frame(frame, dimension):
    amplitude = frame[dimension]
    max_amp = np.max(amplitude)
    norm_frame = amplitude[Z_SLICE] / max_amp

    return norm_frame

# Size of the video (we'll infer it from the first frame)
first_frame = np.load(os.path.join(INPUT_DIR, 'm000000.npy'))
height, width = get_normalised_frame(first_frame, DIMENSION).shape

# Prepare video writer (1920x1080 fallback if needed)
video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),  # codec
    FPS,
    (width, height)
)

# Function to convert matplotlib colormap to BGR image for OpenCV
def to_bgr_image(data_2d):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(data_2d)  # shape (H, W, 4)
    rgb_img = np.delete(rgba_img, 3, 2)  # drop alpha
    bgr_img = (rgb_img[..., ::-1] * 255).astype(np.uint8)
    return bgr_img

# Loop through frames
for i in range(FRAME_COUNT):
    filename = os.path.join(INPUT_DIR, f'm{i:06d}.npy')
    data = np.load(filename)
    frame_bgr = to_bgr_image(get_normalised_frame(data, DIMENSION))
    resized = cv2.resize(frame_bgr, (width, height))
    video_writer.write(resized)

video_writer.release()
print(f'Video saved to {OUTPUT_VIDEO}')

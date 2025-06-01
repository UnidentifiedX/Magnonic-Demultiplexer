import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def get_normalised_frame(frame):
    # amplitude = np.sqrt(frame[0]**2 + frame[1]**2)
    amplitude = frame[0] # mx
    max_amp = np.max(amplitude)
    norm_frame = amplitude[1] / max_amp

    return norm_frame

# Settings
n_frames = 62
input_dir = './main.out'
output_video = 'spin_wave_cv2.mp4'
fps = 10

# Size of the video (we'll infer it from the first frame)
first_frame = np.load(os.path.join(input_dir, 'm000000.npy'))
height, width = get_normalised_frame(first_frame).shape

# Prepare video writer (1920x1080 fallback if needed)
video_writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*'mp4v'),  # codec
    fps,
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
for i in range(n_frames):
    filename = os.path.join(input_dir, f'm{i:06d}.npy')
    data = np.load(filename)
    frame_bgr = to_bgr_image(get_normalised_frame(data))
    resized = cv2.resize(frame_bgr, (width, height))
    video_writer.write(resized)

video_writer.release()
print(f'Video saved to {output_video}')

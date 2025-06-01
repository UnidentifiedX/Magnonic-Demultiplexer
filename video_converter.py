import cv2

# Parameters
image_folder = '.'  # Folder where images are stored
output_file = 'output_video.mp4'
start_idx = 1
end_idx = 41
frame_rate = 10  # FPS

# Generate filename list
filenames = [fr"C:\Sun Zizhuo\Code\mumax\NRP\main.out/m{str(i).zfill(6)}.jpg" for i in range(start_idx, end_idx)]

# Read first image to get size
frame = cv2.imread(filenames[0])
height, width, layers = frame.shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 files
video = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

# Write frames
for fname in filenames:
    img = cv2.imread(fname)
    if img is not None:
        video.write(img)

video.release()
print("Video saved as", output_file)
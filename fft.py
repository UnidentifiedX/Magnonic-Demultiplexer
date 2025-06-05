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
Z_SLICE = 0
DT = 50e-12
FRAME_COUNT = 2401
DIMENSION = Dimension.X

def get_fft(x: tuple, y: tuple, window: tuple, path):
    x_start, x_end = x
    y_start, y_end = y
    signal_start, signal_end = window
    signal = []

    for i in range(FRAME_COUNT):
        data = np.load(os.path.join(INPUT_DIR, f'{path}{i:06d}.npy'))  # shape: (3, Nz, Ny, Nx)
        slice_2d = data[DIMENSION.value, Z_SLICE, y_start:y_end, x_start:x_end]
        avg_val = np.mean(slice_2d)
        signal.append(avg_val)
    
    signal = np.array(signal)

    # Plot the time-domain signal
    plt.plot(signal)
    plt.title(f"Average Magnetisation ({DIMENSION.name} axis)")
    plt.xlabel("Frame")
    plt.ylabel("Magnetisation")
    plt.grid(True)
    plt.show()

    # Extract signal window for FFT
    signal_window = signal[signal_start:signal_end]

    # FFT
    N = len(signal_window)
    windowed_signal = signal_window * np.hanning(N)
    fft = np.fft.rfft(windowed_signal)
    freqs = np.fft.rfftfreq(N, d=DT)
    fft_magnitude = np.abs(fft)

    return freqs, fft_magnitude

# Run FFT
top_waveguide_frequencies, top_waveguide_magnitudes = get_fft(
    (440, 450), (70, 100), (200, FRAME_COUNT), UPPER_WAVEGUIDE_FILE_PREFIX
)

# Plot FFT
plt.figure(figsize=(8, 5))
plt.plot(top_waveguide_frequencies * 1e-9, top_waveguide_magnitudes)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Amplitude")
plt.title("FFT of Spin Wave at Detector")
plt.grid(True)

# Mark expected frequencies
for freq in [2.6, 2.8]:
    plt.axvline(x=freq, color='r', linestyle='--', alpha=0.7, label=f'{freq} GHz')

plt.legend()
plt.tight_layout()
plt.show()

# ======================== SANITY CHECK ========================
# # === Detector box (grid indices) ===
# x_start, x_end = 440, 450
# y_start, y_end = 70, 100

# # === LOAD DATA ===
# data = np.load(os.path.join(INPUT_DIR, f'{UPPER_WAVEGUIDE_FILE_PREFIX}000000.npy'))[Dimension.Z.value, Z_SLICE]

# # === PLOT ===
# plt.figure(figsize=(10, 4))
# plt.imshow(data, cmap='seismic', origin='lower')  # transpose for correct x-y orientation
# plt.colorbar(label='Magnetisation (arb. units)')
# plt.title(f'm[{["x", "y", "z"][Dimension.Z.value]}] at Z-slice {Z_SLICE}')

# # Draw detector box
# plt.axvline(x_start, color='lime', linestyle='--')
# plt.axvline(x_end, color='lime', linestyle='--')
# plt.axhline(y_start, color='lime', linestyle='--')
# plt.axhline(y_end, color='lime', linestyle='--')
# plt.text(x_start, y_end+1, 'Detector Region', color='lime')

# plt.xlabel('X index')
# plt.ylabel('Y index')
# plt.tight_layout()
# plt.show()
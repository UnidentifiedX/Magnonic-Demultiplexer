import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum
import collections

class Dimension(Enum):
    X = 0
    Y = 1
    Z = 2

FILE_PREFIX = "m"
INPUT_DIR = './main.out'
DT = 50e-12
FRAME_COUNT = 2401
DIMENSION = Dimension.X

def get_fft(x: tuple, y: tuple, window: tuple, debug=False):
    x_start, x_end = x
    y_start, y_end = y
    signal_start, signal_end = window
    signal = []

    for i in range(FRAME_COUNT):
        data = np.load(os.path.join(INPUT_DIR, f'{FILE_PREFIX}{i:06d}.npy'))  # shape: (3, Nz, Ny, Nx)
        slice_2d = data[DIMENSION.value, 0, y_start:y_end, x_start:x_end] # 0 becuase there is only one Z layer
        avg_val = np.mean(slice_2d)
        signal.append(avg_val)
    
    signal = np.array(signal)

    if debug:
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
    magnitudes = np.abs(fft)

    if debug:
        # Plot FFT
        plt.figure(figsize=(8, 5))
        plt.plot(freqs * 1e-9, magnitudes)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Amplitude")
        plt.title("FFT of Spin Wave at Detector")
        plt.grid(True)

        # Mark expected frequencies
        for freq in [2.6, 2.8]:
            plt.axvline(x=freq, color='r', linestyle='--', alpha=0.4, label=f'{freq} GHz')

        plt.legend()
        plt.tight_layout()
        plt.show()

    binned_freqs = np.round(freqs / 100e6) * 100e6 # round to nearest 100 MHz
    bin_dict = collections.defaultdict(float)
    for f, mag in zip(binned_freqs, magnitudes):
        bin_dict[f] += mag

    return bin_dict

# Run FFT
top_waveguide_frequencies = get_fft((448, 450), (70, 100), (200, FRAME_COUNT))
bottom_waveguide_frequencies = get_fft((448, 450), (0, 30), (200, FRAME_COUNT))

print(top_waveguide_frequencies[2.6e9], top_waveguide_frequencies[2.8e9])
print(bottom_waveguide_frequencies[2.6e9], bottom_waveguide_frequencies[2.8e9])

# sorted_bins = sorted(top_waveguide_frequencies.items())  # sort by frequency
# bin_freqs, bin_magnitudes = zip(*sorted_bins)

# # Convert to GHz for plotting
# bin_freqs_ghz = np.array(bin_freqs) * 1e-9

# # Plot
# plt.figure(figsize=(8, 4))
# plt.bar(bin_freqs_ghz, bin_magnitudes, width=0.08, align='center', color='skyblue', edgecolor='black')
# plt.xlabel("Frequency (GHz)")
# plt.ylabel("Integrated Amplitude")
# plt.title("Binned FFT Magnitudes (100 MHz bins)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ======================== SANITY CHECK ========================
# # === Detector box (grid indices) ===
# x_start, x_end = 438, 440
# y_start, y_end = 0, 30

# # === LOAD DATA ===
# data = np.load(os.path.join(INPUT_DIR, f'{FILE_PREFIX}000220.npy'))[Dimension.X.value, 0]

# # === PLOT ===
# plt.figure(figsize=(10, 4))
# plt.imshow(data, cmap='seismic', origin='lower')
# plt.colorbar(label='Magnetisation (arb. units)')
# plt.title(f'm[{["x", "y", "z"][Dimension.X.value]}]')

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
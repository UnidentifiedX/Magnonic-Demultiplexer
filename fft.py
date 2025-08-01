import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum
import collections
from mx3_utils import Dimension

def get_fft(x_range: tuple, y_range: tuple, window: tuple, input_dir: str, frame_count: int, dt: int, dimension: 'Dimension', graph_name, normalized=False, debug=False, save_graphs=True):
    x_start, x_end = x_range
    y_start, y_end = y_range
    signal_start, signal_end = window
    signal = []

    for i in range(frame_count):
        data = np.load(os.path.join(input_dir, f'm{i:06d}.npy'))  # shape: (3, Nz, Ny, Nx)
        slice_2d = data[dimension.value, 0, y_start:y_end, x_start:x_end] # 0 becuase there is only one Z layer
        avg_val = np.mean(slice_2d)
        signal.append(avg_val)
    
    signal = np.array(signal)

    if debug or save_graphs:
        # Plot the time-domain signal
        plt.figure(figsize=(10, 5))
        plt.plot(signal)
        plt.title(f"Average Magnetisation ({dimension.name} axis) at {graph_name} Detector")
        plt.xlabel("Frame")
        plt.ylabel("Magnetisation")
        plt.grid(True)
        
        if save_graphs:
            plt.savefig(os.path.join(input_dir, f"../{graph_name}_fft_signal.png"))
        elif debug:
            plt.show()

        plt.close()

    # Extract signal window for FFT
    signal_window = signal[signal_start:signal_end]

    # FFT
    N = len(signal_window)
    windowed_signal = signal_window * np.hanning(N)
    fft = np.fft.rfft(windowed_signal)
    freqs = np.fft.rfftfreq(N, d=dt)
    magnitudes = np.abs(fft)

    if debug or save_graphs:
        # Plot FFT
        plt.figure(figsize=(8, 5))
        plt.plot(freqs * 1e-9, magnitudes)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Amplitude")
        plt.title(f"FFT of Spin Wave at {graph_name} Detector")
        plt.grid(True)

        # Mark expected frequencies
        for freq in [2.6, 2.8]:
            plt.axvline(x=freq, color='r', linestyle='--', alpha=0.4, label=f'{freq} GHz')

        plt.legend()
        plt.tight_layout()

        if save_graphs:
            plt.savefig(os.path.join(input_dir, f"../{graph_name}_fft_plot.png"))
        elif debug:
            plt.show()

        plt.close()

    binned_freqs = np.round(freqs / 100e6) * 100e6 # round to nearest 100 MHz
    bin_dict = collections.defaultdict(float)
    for f, mag in zip(binned_freqs, magnitudes):
        bin_dict[f] += mag

    if normalized:
        total = sum(bin_dict.values())
        if total > 0:
            for f in bin_dict:
                bin_dict[f] /= total

    return bin_dict

if __name__ == "__main__":
    FILE_PREFIX = "m"
    INPUT_DIR = './template.out'
    DT = 50e-12
    FRAME_COUNT = 1001
    DIMENSION = Dimension.X

    print(np.load(os.path.join(INPUT_DIR, f'{FILE_PREFIX}000000.npy')).shape)  # Check shape of the data

    # Run FFT
    top_waveguide_frequencies = get_fft(
        x_range=(224, 225),
        y_range=(35, 50),
        window=(200, FRAME_COUNT),
        input_dir=INPUT_DIR,
        frame_count=FRAME_COUNT,
        dt=DT,
        dimension=DIMENSION,
        graph_name="top_waveguide",
        debug=True,
        save_graphs=False
    )
    bottom_waveguide_frequencies = get_fft(
        x_range=(224, 225),
        y_range=(0, 15),
        window=(200, FRAME_COUNT),
        input_dir=INPUT_DIR,
        frame_count=FRAME_COUNT,
        dt=DT,
        dimension=DIMENSION,
        graph_name="bottom_waveguide",
        save_graphs=False
    )

    print(top_waveguide_frequencies[2.6e9], top_waveguide_frequencies[2.8e9])
    print(bottom_waveguide_frequencies[2.6e9], bottom_waveguide_frequencies[2.8e9])

    # # ======================== SANITY CHECK ========================
    # # === Detector box (grid indices) ===
    # x_start, x_end = 224, 225
    # y_start, y_end = 35, 50

    # # === LOAD DATA ===
    # data = np.load(os.path.join(INPUT_DIR, f'{FILE_PREFIX}001000.npy'))[Dimension.X.value, 0]

    # # === PLOT ===
    # plt.figure(figsize=(10, 4))
    # vabs = np.max(np.abs(data))
    # plt.imshow(data, cmap='seismic', vmin=-vabs, vmax=vabs, origin='lower')
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
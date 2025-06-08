from mx3_utils import Dimension
from fft import get_fft
import os

def evaluate_objective(input_dir, dimension=Dimension.X, frame_count=1201, dt=50e-12):
    top_waveguide_frequencies = get_fft(
        x_range=(224, 225),
        y_range=(35, 50),
        window=(200, frame_count),
        input_dir=input_dir,
        frame_count=frame_count,
        dt=dt,
        dimension=dimension,
        graph_name="top_waveguide",
        save_graphs=True
    )
    bottom_waveguide_frequencies = get_fft(
        x_range=(224, 225),
        y_range=(0, 15),
        window=(200, frame_count),
        input_dir=input_dir,
        frame_count=frame_count,
        dt=dt,
        dimension=dimension,
        graph_name="bottom_waveguide",
        save_graphs=True
    )

    print("Top:    2.6 GHz =", top_waveguide_frequencies[2.6e9], " | 2.8 GHz =", top_waveguide_frequencies[2.8e9])
    print("Bottom: 2.6 GHz =", bottom_waveguide_frequencies[2.6e9], " | 2.8 GHz =", bottom_waveguide_frequencies[2.8e9])


    score = max(top_waveguide_frequencies[2.6e9] - bottom_waveguide_frequencies[2.6e9], 0) * \
        max(bottom_waveguide_frequencies[2.8e9] - top_waveguide_frequencies[2.8e9], 0) 
    # top waveguide 2.6 GHz should be higher than bottom waveguide 2.6 GHz, and bottom waveguide 2.8 GHz should be higher than top waveguide 2.8 GHz

    return score
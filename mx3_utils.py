from io import TextIOWrapper
import subprocess
from enum import Enum
import numpy as np

class Dimension(Enum):
    X = 0
    Y = 1
    Z = 2

def write_log_headers(f: TextIOWrapper):
    f.write("Index,Iteration,Change,Flipped?,Score,Time\n")

def write_log(f: TextIOWrapper, index: int, iteration: int, change: int, flipped: bool, score: float, time: float):
    f.write(f"{index},{iteration},{change},{flipped},{score},{time}\n")

def generate_random_grid(cell_size: int, grid_size: tuple[int, int], choices: list[int]) -> np.ndarray:
    """
    Generates a 2D grid where each block of size (cell_size x cell_size)
    is filled with the same random value from `choices`.

    Parameters:
    - cell_size: size of each square block (e.g., 10)
    - grid_size: full grid size as (height, width) in cells (e.g., (50, 50))
    - choices: list of values to choose from (e.g., [0, 2])

    Returns:
    - A 2D NumPy array of shape grid_size filled with blocks of consistent values
    """
    block_shape = (grid_size[0] // cell_size, grid_size[1] // cell_size)
    block_values = np.random.choice(choices, size=block_shape)
    full_grid = np.kron(block_values, np.ones((cell_size, cell_size), dtype=int))
    return full_grid

def generate_mx3(M, output_path, template_path, X_OFFSET=100, HEIGHT=50):
    """
    Generate a Mumax3 script from the given magnetisation matrix M.
    
    Args:
        M (np.ndarray): Magnetisation matrix.
        output_path (str): Path to save the generated Mumax3 script.
        template_path (str): Path to the Mumax3 template file.
        X_OFFSET (int): X offset for the design region.
        HEIGHT (int): Height of the design region.
    """
    with open(template_path, "r") as template_file:
        content = template_file.read()

    magnetisation = ""
    for i in range(M.shape[0]):     # i = y (down in NumPy)
        for j in range(M.shape[1]): # j = x (right in NumPy)
            region = M[i, j]
            x = X_OFFSET + j
            y = HEIGHT - 1 - i  # flip vertically for Mumax3
            magnetisation += f"DefRegionCell({region}, {x}, {y}, 0)\n"
    
    content = content.replace("// {{ INSERT CELL CODE HERE }}", magnetisation)

    with open(output_path, "w+") as output_file:
        output_file.seek(0)
        output_file.write(content)

def run_mx3(mx3_path, mx3_convert_path, mx3_file_path, output_dir):
    mx3_result = subprocess.run([mx3_path, mx3_file_path], capture_output=True, text=True)
    
    if mx3_result.returncode == 0:
        mx3_convert_result = subprocess.run(
            f"{mx3_convert_path} -numpy {output_dir}/**/*.ovf",
            shell=True, capture_output=True, text=True
        )
    else:
        mx3_convert_result = None
        print(mx3_result.stderr)
        exit(0)

    return {
        'mumax3_stdout': mx3_result.stdout,
        'mumax3_stderr': mx3_result.stderr,
        'mumax3_returncode': mx3_result.returncode,
        'convert_stdout': mx3_convert_result.stdout if mx3_convert_result else '',
        'convert_stderr': mx3_convert_result.stderr if mx3_convert_result else '',
        'convert_returncode': mx3_convert_result.returncode if mx3_convert_result else -1
    }
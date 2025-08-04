from io import TextIOWrapper
import subprocess
from enum import Enum

class Dimension(Enum):
    X = 0
    Y = 1
    Z = 2

def write_log_headers(f: TextIOWrapper):
    f.write("Index,Iteration,Change,Flipped?,Score,Time\n")

def write_log(f: TextIOWrapper, index: int, iteration: int, change: int, flipped: bool, score: float, time: float):
    f.write(f"{index},{iteration},{change},{flipped},{score},{time}\n")

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
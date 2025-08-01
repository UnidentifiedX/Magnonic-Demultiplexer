import numpy as np
import os
import shutil
from mx3_utils import generate_mx3
from dbs import direct_binary_search
from dbs_decay import direct_binary_search_decay

M0 = np.random.choice([0, 1], size=(50, 50)) * 2 # 2 because design region area is 2
# LENGTH = 500
# HEIGHT = 100
# X_OFFSET = 200
MX3_PATH = 'mumax3'
MX3_CONVERT_PATH = 'mumax3-convert'
TEMPLATE_PATH = './template.mx3'
OUTPUT_DIR = './output'
FRAME_COUNT = 1001
MAX_ITERATIONS = 10
TOLERANCE = 0.01
INITIAL_PATCH_SIZE = 10
PATCH_DECAY_INTERVAL = 1

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)  # Clear output directory if it exists
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory

# M_output, final_score = direct_binary_search(initial_M=M0,
#                                              mx3_path=MX3_PATH,
#                                              mx3_convert_path=MX3_CONVERT_PATH,
#                                              template_path=TEMPLATE_PATH,
#                                              output_path=OUTPUT_DIR,
#                                              frame_count=FRAME_COUNT,
#                                              max_iterations=MAX_ITERATIONS,
#                                              tolerance=TOLERANCE)
M_output, final_score = direct_binary_search_decay(
    initial_M=M0,
    mx3_path=MX3_PATH,
    mx3_convert_path=MX3_CONVERT_PATH,
    template_path=TEMPLATE_PATH,
    output_path=OUTPUT_DIR,
    frame_count=FRAME_COUNT,
    max_iterations=MAX_ITERATIONS,
    tolerance=TOLERANCE,
    initial_patch_size=INITIAL_PATCH_SIZE,  # Initial patch size for decay
    patch_decay_interval=PATCH_DECAY_INTERVAL   # Decay interval for patch size
)
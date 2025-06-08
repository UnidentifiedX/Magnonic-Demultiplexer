import numpy as np
import os
import shutil
from mx3_utils import generate_mx3
from dbs import direct_binary_search

M0 = np.random.choice([0, 1], size=(50, 50)) * 2 # 2 because design region area is 2
# LENGTH = 500
# HEIGHT = 100
# X_OFFSET = 200
OUTPUT_DIR = './output'

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)  # Clear output directory if it exists
os.mkdir(OUTPUT_DIR)

M_output, final_score = direct_binary_search(initial_M=M0, 
                                             template_path='./template.mx3',
                                             output_path=OUTPUT_DIR,
                                             frame_count=1001,
                                             max_iterations=2,
                                             tolerance=0.01)
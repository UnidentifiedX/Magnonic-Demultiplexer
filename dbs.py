import numpy as np
import os
from mx3_utils import generate_mx3, run_mx3
from objective_function import evaluate_objective
import time
import shutil

MAGNETISATION_FLIP = 2  # The value to flip the magnetisation, assuming design region area is 2

def direct_binary_search(initial_M, mx3_path, mx3_convert_path, template_path, output_path, frame_count, max_iterations=1000, tolerance=0.01):
    M = initial_M.copy()
    best_score = 0
    
    for i in range(max_iterations):
        improvement = False
        coords = [(i, j) for i in range(M.shape[0]) for j in range(M.shape[1])]
        np.random.shuffle(coords)

        run_folder = os.path.join(output_path, f"{i:06d}")
        os.mkdir(run_folder)

        for j, (y, x) in enumerate(coords):
            start = time.time()

            output_folder = os.path.join(run_folder, f"{j:06d}")
            os.mkdir(output_folder)
            file_path = os.path.join(output_folder, f"{j:06d}.mx3")

            M[y, x] = MAGNETISATION_FLIP - M[y, x]  # Flip the magnetisation at (y, x), 2 because design region area is 2
            generate_mx3(M, file_path, template_path=template_path)
            output = run_mx3(mx3_path=mx3_path, mx3_convert_path=mx3_convert_path, mx3_file_path=file_path, output_dir=output_folder)
            score = evaluate_objective(f"{output_folder}/{j:06d}.out", frame_count=frame_count)

            if score > 0 and (best_score == 0 or (score - best_score) / best_score > tolerance):  # Only accept if score is positive and improves
                print(f"[Iteration {i} change {j}] Accepted flip at ({y}, {x}), score: {score}, time taken: {time.time() - start:.2f}s; Difference: {score - best_score}")
                best_score = score
                improvement = True
            else:
                print(f"[Iteration {i} change {j}] Rejected flip at ({y}, {x}), score: {score}, time taken: {time.time() - start:.2f}s")
                M[y, x] = MAGNETISATION_FLIP - M[y, x] # Revert the flip if no improvement, 2 because design region area is 2
                shutil.rmtree(output_folder)  # Clean up the output folder

        if not improvement:
            print(f"No improvement in iteration {i}, stopping search.")
            break

    return M, best_score
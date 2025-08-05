import numpy as np
import os
import mx3_utils
from objective_function import evaluate_objective
import time
import shutil
import glob

MAGNETISATION_FLIP = 2  # The value to flip the magnetisation, assuming design region area is 2
FLUSH_INTERVAL = 50  # Interval to flush the output file

def get_patch_coords(M, patch_size):
    h, w = M
    return [(i, j) for i in range(0, h - patch_size + 1) for j in range(0, w - patch_size + 1)]

def flip_patch(M, y, x, patch_size):
    M[y:y+patch_size, x:x+patch_size] = MAGNETISATION_FLIP - M[y:y+patch_size, x:x+patch_size]
    return M

def direct_binary_search_decay(initial_M, mx3_path, mx3_convert_path, template_path, output_path, frame_count, max_iterations=10, tolerance=0.01, initial_patch_size=10, patch_decay_interval=1):
    M = initial_M.copy()
    best_score = 0

    with open(os.path.join(output_path, 'scores.csv'), 'w') as f:
        mx3_utils.write_log_headers()
        index = 0
        patch_size = initial_patch_size

        for i in range(max_iterations):
            improvement = False
            # coords = [(i, j) for i in range(M.shape[0]) for j in range(M.shape[1])]
            coords = get_patch_coords(M.shape, patch_size)
            np.random.shuffle(coords)

            run_folder = os.path.join(output_path, f"{i:06d}")
            os.mkdir(run_folder)

            for j, (y, x) in enumerate(coords):
                start = time.time()

                output_folder = os.path.join(run_folder, f"{j:06d}")
                os.mkdir(output_folder)
                file_path = os.path.join(output_folder, f"{j:06d}.mx3")

                M = flip_patch(M, y, x, patch_size)  # Flip the magnetisation at (y, x)
                mx3_utils.generate_mx3(M, file_path, template_path=template_path)
                output = mx3_utils.run_mx3(mx3_path=mx3_path, mx3_convert_path=mx3_convert_path, mx3_file_path=file_path, output_dir=output_folder)
                score = evaluate_objective(f"{output_folder}/{j:06d}.out", frame_count=frame_count)
                flipped = False

                if score > 0 and (best_score == 0 or (score - best_score) / best_score > tolerance):  # Only accept if score is positive and improves
                    print(f"[Iteration {i} change {j}] Accepted flip at ({y}, {x}), score: {score}, time taken: {time.time() - start:.2f}s; Difference: {score - best_score}; Patch size: {patch_size}")
                    best_score = score
                    improvement = True
                    flipped = True

                    for file in glob.glob(f"{output_folder}/{j:06d}.out/*.jpg"):
                        shutil.move(file, output_folder)
                    shutil.rmtree(f"{output_folder}/{j:06d}.out")
                else:
                    print(f"[Iteration {i} change {j}] Rejected flip at ({y}, {x}), score: {score}, time taken: {time.time() - start:.2f}s; Patch size: {patch_size}")
                    M = flip_patch(M, y, x, patch_size) # Revert the flip if no improvement
                    shutil.rmtree(output_folder)  # Clean up the output folder

                mx3_utils.write_log(f=f,
                                    index=index, 
                                    iteration=i, 
                                    change=j,
                                    flipped=flipped,
                                    score=score,
                                    time=time.time() - start)

                if j % FLUSH_INTERVAL == 0:
                    f.flush()

            if not improvement:
                if patch_size > 1:
                    patch_size -= 1  # Reduce patch size if no improvement
                else:
                    print(f"No improvement in iteration {i}, stopping search.")
                    f.flush()
                    break

    return M, best_score
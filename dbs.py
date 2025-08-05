import numpy as np
import os
import mx3_utils
from objective_function import evaluate_objective
import time
import shutil
import glob

MAGNETISATION_FLIP = 2
FLUSH_INTERVAL = 50
BLOCK_SIZE = 5  # Size of square block to flip

def direct_binary_search(initial_M, mx3_path, mx3_convert_path, template_path, output_path, frame_count, max_iterations=1000, tolerance=0.01):
    M = initial_M.copy()
    best_score = 0
    index = 0
    height, width = M.shape

    with open(os.path.join(output_path, 'scores.csv'), 'w') as f:
        mx3_utils.write_log_headers(f)

        for i in range(max_iterations):
            improvement = False

            # Generate all valid top-left block coordinates
            coords = [(y, x) for y in range(0, height - BLOCK_SIZE + 1, BLOCK_SIZE)
                            for x in range(0, width - BLOCK_SIZE + 1, BLOCK_SIZE)]
            np.random.shuffle(coords)

            run_folder = os.path.join(output_path, f"{i:06d}")
            os.mkdir(run_folder)

            for j, (y, x) in enumerate(coords):
                start = time.time()

                output_folder = os.path.join(run_folder, f"{j:06d}")
                os.mkdir(output_folder)
                file_path = os.path.join(output_folder, f"{j:06d}.mx3")

                # Flip the entire block
                M[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = MAGNETISATION_FLIP - M[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
                mx3_utils.generate_mx3(M, file_path, template_path=template_path)
                output = mx3_utils.run_mx3(mx3_path=mx3_path, mx3_convert_path=mx3_convert_path, mx3_file_path=file_path, output_dir=output_folder)
                score = evaluate_objective(f"{output_folder}/{j:06d}.out", frame_count=frame_count)
                flipped = False

                if score > 0 and (best_score == 0 or (score - best_score) / best_score > tolerance):
                    print(f"[Iteration {i} change {j}] Accepted block flip at ({y}, {x}), score: {score}, time: {time.time() - start:.2f}s")
                    best_score = score
                    improvement = True
                    flipped = True

                    for file in glob.glob(f"{output_folder}/{j:06d}.out/*.jpg"):
                        shutil.move(file, output_folder)
                    shutil.rmtree(f"{output_folder}/{j:06d}.out")
                else:
                    print(f"[Iteration {i} change {j}] Rejected block flip at ({y}, {x}), score: {score}, time: {time.time() - start:.2f}s")
                    M[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = MAGNETISATION_FLIP - M[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]  # Revert
                    shutil.rmtree(output_folder)
                
                mx3_utils.write_log(f=f,
                                    index=index, 
                                    iteration=i, 
                                    change=j,
                                    flipped=flipped,
                                    score=score,
                                    time=time.time() - start)

                if j % FLUSH_INTERVAL == 0:
                    f.flush()

                index += 1

            if not improvement:
                print(f"No improvement in iteration {i}, stopping search.")
                f.flush()
                break

    return M, best_score
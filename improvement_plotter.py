import pandas as pd
import matplotlib.pyplot as plt
import os

SCORES_DIR = 'scores'

plt.figure(figsize=(10, 6))

for file in os.listdir(SCORES_DIR):
    if file.endswith('.csv'):
        path = os.path.join(SCORES_DIR, file)
        df = pd.read_csv(path)

        # Ensure correct types
        df['Flipped?'] = df['Flipped?'].astype(bool)

        # Filter flipped entries
        flipped_df = df[df['Flipped?']]

        # Use the global index column (assume it's called 'Index')
        plt.plot(flipped_df['Index'], flipped_df['Score'], label=file, marker='o')

# Finalize plot
plt.title('Score Progression (Flipped Only)')
plt.xlabel('Total Change Index')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
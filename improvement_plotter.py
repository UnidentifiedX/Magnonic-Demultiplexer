import pandas as pd
import matplotlib.pyplot as plt

PATH = 'scores.csv'
CHANGES_PER_ITERATION = 2500

df = pd.read_csv(PATH)
df['Flipped?'] = df['Flipped?'].astype(bool)

df['GlobalChange'] = df['Iteration'] * CHANGES_PER_ITERATION + df['Change']
flipped_df = df[df['Flipped?']]

plt.figure(figsize=(10, 6))
plt.plot(flipped_df['GlobalChange'], flipped_df['Score'], label='Score', marker='o')

plt.title('Score Progression (Only Flipped Changes, Global Change Index)')
plt.xlabel('Global Change Number')
plt.ylabel('Score')
plt.grid(True)
plt.tight_layout()
plt.show()
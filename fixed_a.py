"""
Problem 1 fix: two corrections applied together.

CORRECTION A — Exclude the target keyword from n_trends_activated.
  Each image's best_match keyword is the one that placed it in the
  PERSISTENT or FADED group. Counting that keyword in n_trends_activated
  inflates scores for images that already scored high on one keyword
  (they were put in the group precisely because of that score).
  We recompute n_trends_activated excluding best_match each time.

CORRECTION B — Aggregate to trend-season level before any comparison.
  The correct statistical unit is a (keyword × season) cell, not an image.
  We have 10 keywords × 6 seasons = 60 cells.
  We compare the 18 PERSISTENT cells (3 keywords × 6 seasons) against
  the 36 FADED cells (6 keywords × 6 seasons).
  We use a permutation test (10,000 permutations) rather than a t-test
  because n=54 cells is too small to trust parametric normality assumptions.
"""

import pandas as pd
import numpy as np
from scipy import stats

# ── load data ─────────────────────────────────────────────────────────────────
# Load the tau=0.22 similarity matrix (the version with keyword scores intact)
sim_df = pd.read_csv('similarity_matrix22.csv')

keywords = [
    'creamy yellow', 'hot pants', 'moccasins', 'bubble hem',
    'wide leg trousers', 'Y2K fashion', 'thong sandals',
    'strapless dress', 'barrel leg jeans', 'ballet flat shoes'
]

persistence = {
    'hot pants':         'PERSISTENT',
    'moccasins':         'PERSISTENT',
    'wide leg trousers': 'PERSISTENT',
    'strapless dress':   'SPIKE_FADE',
    'creamy yellow':     'LOW_INTEREST',
    'bubble hem':        'LOW_INTEREST',
    'Y2K fashion':       'LOW_INTEREST',
    'thong sandals':     'LOW_INTEREST',
    'barrel leg jeans':  'LOW_INTEREST',
    'ballet flat shoes': 'LOW_INTEREST',
}

THRESHOLD = 0.22

# ── CORRECTION A: exclude target keyword from activation count ─────────────────
# For each image, n_trends_activated counts keywords above threshold
# EXCLUDING the image's own best_match keyword (to break circularity).

def activation_excluding_target(row, keywords, threshold):
    target = row['best_match']
    other_keywords = [k for k in keywords if k != target]
    return sum(row[k] > threshold for k in other_keywords)

print("Computing circularity-corrected n_trends_activated...")
sim_df['n_activated_corrected'] = sim_df.apply(
    activation_excluding_target, axis=1,
    keywords=keywords, threshold=THRESHOLD
)

# Quick sanity check: compare original vs corrected
print(f"Original  n_trends_activated  mean: {sim_df['n_trends_activated'].mean():.3f}")
print(f"Corrected n_trends_activated  mean: {sim_df['n_activated_corrected'].mean():.3f}")
print(f"Difference (bias from circularity): "
      f"{sim_df['n_trends_activated'].mean() - sim_df['n_activated_corrected'].mean():.3f}")

# ── CORRECTION B: aggregate to trend-season cells ─────────────────────────────
# Statistical unit = one (keyword, season) combination.
# We compute the mean corrected activation for each cell.

sim_df['persistence'] = sim_df['best_match'].map(persistence)

# Aggregate: mean corrected activation per keyword × season
cell_df = (
    sim_df
    .groupby(['best_match', 'show'])['n_activated_corrected']
    .agg(mean='mean', std='std', n='count')
    .reset_index()
)
cell_df['persistence'] = cell_df['best_match'].map(persistence)

print(f"\nTotal cells: {len(cell_df)}")
print(cell_df.groupby('persistence')['mean'].describe().round(3))

# Print the cells sorted by their image counts to see the absolute minimum
print(cell_df[['best_match', 'show', 'n']].sort_values(by='n').head(10))

# ── Permutation test on aggregated cells ─────────────────────────────────────
# We compare PERSISTENT cells vs LOW_INTEREST cells.
# Permutation test: shuffle persistence labels 10,000 times,
# recompute the mean difference each time, then see where the
# observed difference falls in that null distribution.

p_cells = cell_df[cell_df['persistence']=='PERSISTENT']['mean'].values
f_cells = cell_df[cell_df['persistence']=='LOW_INTEREST']['mean'].values

observed_diff = p_cells.mean() - f_cells.mean()
print(f"\nObserved mean difference (PERSISTENT - LOW_INTEREST): {observed_diff:.4f}")

# Permutation test
np.random.seed(42)
N_PERM = 10_000
all_values = np.concatenate([p_cells, f_cells])
n_p = len(p_cells)
perm_diffs = np.array([
    np.random.permutation(all_values)[:n_p].mean() -
    np.random.permutation(all_values)[n_p:].mean()
    for _ in range(N_PERM)
])

p_perm = (np.abs(perm_diffs) >= np.abs(observed_diff)).mean()

# Effect size on aggregated cells (Cohen's d on cell means)
pooled_std = np.sqrt(
    (p_cells.std(ddof=1)**2 + f_cells.std(ddof=1)**2) / 2
)
d_cells = observed_diff / pooled_std if pooled_std > 0 else 0

print(f"\n=== CORRECTED Analysis A (trend-season cells) ===")
print(f"PERSISTENT  n_cells={len(p_cells)}  mean={p_cells.mean():.3f}  "
      f"sd={p_cells.std(ddof=1):.3f}")
print(f"LOW_INTEREST n_cells={len(f_cells)}  mean={f_cells.mean():.3f}  "
      f"sd={f_cells.std(ddof=1):.3f}")
print(f"Observed difference: {observed_diff:.4f}")
print(f"Permutation p-value (two-tailed, {N_PERM} permutations): {p_perm:.4f}")
print(f"Cohen's d (cell-level): {d_cells:.3f}")

# Save cell-level data for reporting
cell_df.to_csv('analysis_a_cells.csv', index=False)

# ── Visualise the permutation null distribution ───────────────────────────────
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(perm_diffs, bins=60, color='#C8B8E8', edgecolor='white',
        label='Null distribution (permuted)')
ax.axvline(observed_diff, color='#4A2D8F', linewidth=2.5,
           label=f'Observed Δ = {observed_diff:.4f}')
ax.axvline(-observed_diff, color='#4A2D8F', linewidth=2.5, linestyle='--')
ax.set_xlabel('Mean difference (PERSISTENT − LOW_INTEREST)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Permutation null distribution — corrected Analysis A\n'
             f'(trend–season cells, excluding target keyword)',
             fontsize=11)
ax.legend(fontsize=9)
ax.text(0.97, 0.95,
        f'p = {p_perm:.3f}\nd = {d_cells:.3f}',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=10, color='#4A2D8F',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#EDE9FA', alpha=0.8))
plt.tight_layout()
plt.savefig('fig_permutation_test.png', dpi=180, bbox_inches='tight')
plt.show()
print("\nPermutation plot saved.")

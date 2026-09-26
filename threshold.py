"""
Problem 2 fix: compute Cohen's d and permutation p-value across the full
threshold range tau in [0.18, 0.28] at steps of 0.005.

We apply the corrected statistical unit from Problem 1 fix throughout:
  - Exclude target keyword from activation count
  - Aggregate to trend-season cells
  - Use permutation test for p-value
  - Bootstrap confidence intervals for d

This produces a threshold sensitivity curve that makes the
threshold-dependence of the finding explicit and transparent.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── load data ─────────────────────────────────────────────────────────────────
# We need the raw similarity scores (not the thresholded version)
# so we load the file that still has all keyword columns intact.
sim_df = pd.read_csv('similarity_matrix22.csv')  # keyword scores are the same in all threshold files

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

sim_df['persistence'] = sim_df['best_match'].map(persistence)

# ── threshold sweep ───────────────────────────────────────────────────────────
thresholds = np.arange(0.18, 0.285, 0.005)
N_PERM     = 5000
N_BOOT     = 2000
np.random.seed(42)

results = []

for tau in thresholds:

    # 1. Recompute corrected activation at this threshold
    def corrected_activation(row):
        target = row['best_match']
        others = [k for k in keywords if k != target]
        return sum(row[k] > tau for k in others)

    sim_df['n_act'] = sim_df.apply(corrected_activation, axis=1)

    # 2. Aggregate to trend-season cells
    cell_df = (
        sim_df
        .groupby(['best_match', 'show'])['n_act']
        .mean()
        .reset_index(name='mean_act')
    )
    cell_df['persistence'] = cell_df['best_match'].map(persistence)

    p_vals = cell_df[cell_df['persistence']=='PERSISTENT']['mean_act'].values
    f_vals = cell_df[cell_df['persistence']=='LOW_INTEREST']['mean_act'].values

    # Skip if either group is empty or has zero variance
    if len(p_vals) < 2 or len(f_vals) < 2:
        continue

    obs_diff = p_vals.mean() - f_vals.mean()
    pooled   = np.sqrt((p_vals.std(ddof=1)**2 + f_vals.std(ddof=1)**2) / 2)
    d_obs    = obs_diff / pooled if pooled > 0 else 0

    # 3. Permutation p-value
    combined = np.concatenate([p_vals, f_vals])
    n_p      = len(p_vals)
    perm_d   = np.array([
        (lambda s: (s[:n_p].mean()-s[n_p:].mean()) /
         (np.sqrt((s[:n_p].std(ddof=1)**2+s[n_p:].std(ddof=1)**2)/2) or 1))
        (np.random.permutation(combined))
        for _ in range(N_PERM)
    ])
    p_perm = (np.abs(perm_d) >= np.abs(d_obs)).mean()

    # 4. Bootstrap 95% CI for d
    boot_d = []
    for _ in range(N_BOOT):
        bp = np.random.choice(p_vals, size=len(p_vals), replace=True)
        bf = np.random.choice(f_vals, size=len(f_vals), replace=True)
        pool_b = np.sqrt((bp.std(ddof=1)**2 + bf.std(ddof=1)**2) / 2)
        boot_d.append((bp.mean()-bf.mean()) / pool_b if pool_b > 0 else 0)

    ci_lo, ci_hi = np.percentile(boot_d, [2.5, 97.5])

    results.append({
        'tau':       round(tau, 3),
        'mean_p':    p_vals.mean(),
        'mean_f':    f_vals.mean(),
        'abs_diff':  obs_diff,
        'cohens_d':  d_obs,
        'ci_lo':     ci_lo,
        'ci_hi':     ci_hi,
        'p_perm':    p_perm,
        'n_p_cells': len(p_vals),
        'n_f_cells': len(f_vals),
    })
    print(f"τ={tau:.3f}  d={d_obs:.3f} [{ci_lo:.3f},{ci_hi:.3f}]  "
          f"p_perm={p_perm:.3f}  abs_diff={obs_diff:.4f}")

res_df = pd.DataFrame(results)
res_df.to_csv('threshold_sensitivity_curve.csv', index=False)
print("\nSaved threshold_sensitivity_curve.csv")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# Panel A: Cohen's d with 95% CI
ax = axes[0]
ax.fill_between(res_df['tau'], res_df['ci_lo'], res_df['ci_hi'],
                color='#C8B8E8', alpha=0.4, label='95% bootstrap CI')
ax.plot(res_df['tau'], res_df['cohens_d'],
        color='#4A2D8F', linewidth=2.5, marker='o', markersize=5,
        label="Cohen's d (cell-level)")
ax.axhline(0,    color='black',  linewidth=0.8, linestyle='--')
ax.axhline(0.2,  color='gray',   linewidth=0.8, linestyle=':',  label='Small (d=0.2)')
ax.axhline(0.5,  color='gray',   linewidth=0.8, linestyle='-.', label='Medium (d=0.5)')
ax.axvline(0.22, color='#0F6E6E',linewidth=1.5, linestyle='--', label='τ=0.22 (selected)')
ax.axvline(0.26, color='#B85C00',linewidth=1.5, linestyle='--', label='τ=0.26 (strict)')
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("Threshold sensitivity: effect size (corrected statistical unit)",
             fontsize=11)
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.set_ylim(-0.5, 1.2)

# Panel B: absolute mean difference and permutation p-value
ax2 = axes[1]
color_diff = '#4A2D8F'
color_p    = '#C0392B'
ax2.plot(res_df['tau'], res_df['abs_diff'],
         color=color_diff, linewidth=2.5, marker='o', markersize=5,
         label='Mean difference (PERSISTENT − LOW_INTEREST)')
ax2.set_ylabel('Absolute mean difference\n(keywords activated)', fontsize=10,
               color=color_diff)
ax2.tick_params(axis='y', labelcolor=color_diff)

ax2b = ax2.twinx()
ax2b.plot(res_df['tau'], res_df['p_perm'],
          color=color_p, linewidth=2, linestyle='--', marker='s', markersize=4,
          label='Permutation p-value')
ax2b.axhline(0.05, color=color_p, linewidth=0.8, linestyle=':',
             label='α = 0.05')
ax2b.set_ylabel('Permutation p-value', fontsize=10, color=color_p)
ax2b.tick_params(axis='y', labelcolor=color_p)
ax2b.set_ylim(0, 0.6)

ax2.set_xlabel('Cosine similarity threshold τ', fontsize=11)
ax2.set_title('Absolute difference and permutation p-value across thresholds',
              fontsize=11)

lines1, labs1 = ax2.get_legend_handles_labels()
lines2, labs2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1+lines2, labs1+labs2, fontsize=8, loc='upper left')
ax2.axvline(0.22, color='#0F6E6E', linewidth=1.5, linestyle='--')
ax2.axvline(0.26, color='#B85C00', linewidth=1.5, linestyle='--')

plt.tight_layout()
plt.savefig('fig_threshold_sensitivity.png', dpi=180, bbox_inches='tight')
plt.show()
print("Figure saved: fig_threshold_sensitivity.png")

# ── Print summary table for paper ─────────────────────────────────────────────
print("\n=== Summary table for paper ===")
print(res_df[['tau','mean_p','mean_f','abs_diff','cohens_d','ci_lo','ci_hi','p_perm']]
      .to_string(index=False, float_format='%.3f'))

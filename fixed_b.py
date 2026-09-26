"""
Problem 4 fix: apply Bonferroni and Benjamini-Hochberg corrections
to the 10 OLS regression p-values from Analysis B.

With 6 data points per regression (one per season), OLS p-values are
already unreliable — the t-distribution with 4 degrees of freedom has
very heavy tails. Additionally, running 10 tests at alpha=0.05 produces
an expected false positive rate of 1-(0.95^10) ≈ 40%.

We apply two standard corrections:
  Bonferroni:          p_corrected = min(p_raw * n_tests, 1.0)
                       threshold = 0.05 / 10 = 0.005
  Benjamini-Hochberg:  controls false discovery rate at 5%
                       less conservative, appropriate for exploratory work

We also add bootstrap confidence intervals for each slope (β) to show
the uncertainty of a slope estimated from only 6 points.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── load similarity matrix ────────────────────────────────────────────────────
sim_df = pd.read_csv('similarity_matrix22.csv')

keywords = [
    'creamy yellow', 'hot pants', 'moccasins', 'bubble hem',
    'wide leg trousers', 'Y2K fashion', 'thong sandals',
    'strapless dress', 'barrel leg jeans', 'ballet flat shoes'
]

# Season ordering — must be explicit and oldest-first
season_order = {
    'Spring 2024 Ready-to-Wear': 0,
    'Fall 2024 Ready-to-Wear':   1,
    'Spring 2025 Ready-to-Wear': 2,
    'Fall 2025 Ready-to-Wear':   3,
    'Spring 2026 Ready-to-Wear': 4,
    'Fall 2026 Ready-to-Wear':   5,
}
sim_df['season_idx'] = sim_df['show'].map(season_order)

# ── OLS regression per keyword ────────────────────────────────────────────────
regression_results = []
N_BOOT = 5000
np.random.seed(42)

for kw in keywords:
    # Per-season mean similarity for this keyword
    season_means = (
        sim_df.groupby('season_idx')[kw]
        .mean()
        .reset_index()
        .sort_values('season_idx')
    )
    x = season_means['season_idx'].values.astype(float)
    y = season_means[kw].values

    n = len(x)  # = 6

    # OLS regression
    slope, intercept, r, p_raw, se = stats.linregress(x, y)
    r_sq = r**2

    # Bootstrap CI for slope (resampling season means with replacement)
    boot_slopes = []
    for _ in range(N_BOOT):
        idx = np.random.choice(n, size=n, replace=True)
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 2:
            continue
        bs, *_ = stats.linregress(xb, yb)
        boot_slopes.append(bs)
    ci_lo, ci_hi = np.percentile(boot_slopes, [2.5, 97.5])

    regression_results.append({
        'keyword': kw,
        'beta':    slope,
        'r2':      r_sq,
        'p_raw':   p_raw,
        'se':      se,
        'ci_lo':   ci_lo,
        'ci_hi':   ci_hi,
        'n_seasons': n,
    })

res_df = pd.DataFrame(regression_results).sort_values('beta', ascending=False)

# ── Multiple testing corrections ──────────────────────────────────────────────
p_raw_arr = res_df['p_raw'].values

# Bonferroni
_, p_bonf, _, _ = multipletests(p_raw_arr, alpha=0.05, method='bonferroni')

# Benjamini-Hochberg (False Discovery Rate)
_, p_bh, _, _ = multipletests(p_raw_arr, alpha=0.05, method='fdr_bh')

res_df['p_bonferroni'] = p_bonf
res_df['p_bh']         = p_bh
res_df['sig_bonf']     = p_bonf < 0.05
res_df['sig_bh']       = p_bh < 0.05

# ── Print results table ───────────────────────────────────────────────────────
print("=== Analysis B — OLS regressions with multiple testing corrections ===")
print(f"{'Keyword':<22} {'β':>9} {'R²':>6} {'p_raw':>7} "
      f"{'p_Bonf':>8} {'p_BH':>7} {'CI_lo':>7} {'CI_hi':>7} "
      f"{'Sig_BH':>7}")
print("-" * 90)
for _, row in res_df.iterrows():
    sig = "YES" if row['sig_bh'] else "no"
    print(f"{row['keyword']:<22} {row['beta']:>9.5f} {row['r2']:>6.3f} "
          f"{row['p_raw']:>7.3f} {row['p_bonferroni']:>8.3f} "
          f"{row['p_bh']:>7.3f} {row['ci_lo']:>7.5f} {row['ci_hi']:>7.5f} "
          f"{sig:>7}")

print(f"\nBonferroni threshold: p < {0.05/len(keywords):.4f}")
print(f"Keywords surviving Bonferroni: {res_df['sig_bonf'].sum()}")
print(f"Keywords surviving BH (FDR 5%): {res_df['sig_bh'].sum()}")

# Save
res_df.to_csv('analysis_b_corrected.csv', index=False)

# ── Plot: slope + CI forest plot ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

colors = {
    'PERSISTENT':  '#0F6E6E',
    'LOW_INTEREST':'#888899',
    'SPIKE_FADE':  '#C0392B',
}
persistence_map = {
    'hot pants':'PERSISTENT', 'moccasins':'PERSISTENT',
    'wide leg trousers':'PERSISTENT', 'strapless dress':'SPIKE_FADE',
    'creamy yellow':'LOW_INTEREST', 'bubble hem':'LOW_INTEREST',
    'Y2K fashion':'LOW_INTEREST', 'thong sandals':'LOW_INTEREST',
    'barrel leg jeans':'LOW_INTEREST', 'ballet flat shoes':'LOW_INTEREST',
}

# Sort by slope for readability
res_sorted = res_df.sort_values('beta')
y_pos = np.arange(len(res_sorted))

for i, (_, row) in enumerate(res_sorted.iterrows()):
    col = colors[persistence_map[row['keyword']]]
    # CI bar
    ax.barh(i, row['ci_hi'] - row['ci_lo'],
            left=row['ci_lo'], height=0.5,
            color=col, alpha=0.25)
    # Point estimate
    ax.plot(row['beta'], i, 'o', color=col, markersize=7, zorder=3)
    # Mark BH-significant with star
    if row['sig_bh']:
        ax.text(row['ci_hi'] + 0.00003, i, '★ BH sig.',
                va='center', fontsize=8, color=col)

ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.set_yticks(y_pos)
ax.set_yticklabels(res_sorted['keyword'], fontsize=9)
ax.set_xlabel('OLS slope β (change in mean CLIP similarity per season)',
              fontsize=10)
ax.set_title('Analysis B — Seasonal slopes with 95% bootstrap CI\n'
             'after Bonferroni and Benjamini-Hochberg correction',
             fontsize=10)

# Legend
patches = [
    mpatches.Patch(color='#0F6E6E', label='PERSISTENT'),
    mpatches.Patch(color='#C0392B', label='SPIKE_FADE'),
    mpatches.Patch(color='#888899', label='LOW_INTEREST'),
]
ax.legend(handles=patches, fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig('fig_analysis_b_corrected.png', dpi=180, bbox_inches='tight')
plt.show()
print("\nForest plot saved: fig_analysis_b_corrected.png")

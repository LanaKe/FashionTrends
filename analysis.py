import pandas as pd
from scipy import stats
import numpy as np

sim_df = pd.read_csv('similarity_matrix.csv')

# Assign persistence labels — update based on your actual curve inspection
persistence = {
    'creamy yellow':  'faded',
    'hot pants':    'persistent',
    'ballet flat shoes':     'faded',
    'wide leg trousers':     'persistent',
    'moccasins':       'persistent',
    'bubble hem':        'faded',
    'Y2K fashion':           'faded',
    'thong sandals':   'faded',
    'barrel leg jeans':      'faded',
    'strapless dress':    'spike_fade',
}
sim_df['persistence'] = sim_df['best_match'].map(persistence)


# Primary comparison: persistent vs spike_fade
p_group = sim_df[sim_df['persistence']=='persistent']['n_trends_activated']
s_group = sim_df[sim_df['persistence']=='faded']['n_trends_activated']
#s_group = sim_df[sim_df['persistence']=='spike_fade']['n_trends_activated']


t, p = stats.ttest_ind(p_group, s_group, equal_var=False)  # Welch's t-test
d = (p_group.mean() - s_group.mean()) / np.sqrt(
    (p_group.std()**2 + s_group.std()**2) / 2)  # Cohen's d effect size

print('=== Analysis A: Cross-trend activation ===')
print(f'Persistent  — mean: {p_group.mean():.3f}, std: {p_group.std():.3f}, n={len(p_group)}')
print(f'Faded  — mean: {s_group.mean():.3f}, std: {s_group.std():.3f}, n={len(s_group)}')
print(f'Welch t={t:.3f}, p={p:.4f}')
print(f"Cohen's d effect size: {d:.3f}")
print(f'Result: {"SIGNIFICANT (p<0.05)" if p<0.05 else "NOT SIGNIFICANT"}')

import pandas as pd
import matplotlib.pyplot as plt

sim_df = pd.read_csv('similarity_matrix.csv')



# Season ordering — ordinal for plotting
season_order = {
    'Spring 2024 Ready-to-Wear': 0,
    'Fall 2024 Ready-to-Wear':   1,
    'Spring 2025 Ready-to-Wear': 2,
    'Fall 2025 Ready-to-Wear':   3,
    'Spring 2026 Ready-to-Wear': 4,
    'Fall 2026 Ready-to-Wear':   5,
}
sim_df['season_idx'] = sim_df['show'].map(season_order)

keywords = [c for c in sim_df.columns
            if c not in ['designer','show',
                         'best_match','best_score','n_trends_activated',
                         'persistence','season_idx']]

# Per-season mean similarity for each keyword
season_trend = sim_df.groupby('season_idx')[keywords].mean()

# Compute slope (linear trend) for each keyword across seasons
from scipy import stats as sp
slopes = {}
for kw in keywords:
    x = season_trend.index.values
    y = season_trend[kw].values
    slope, _, r, p, _ = sp.linregress(x, y)
    slopes[kw] = {'slope': slope, 'r_squared': r**2, 'p_value': p}

slopes_df = pd.DataFrame(slopes).T.sort_values('slope', ascending=False)
print(slopes_df.round(5))
slopes_df.to_csv('keyword_slopes.csv')

import pandas as pd
from scipy import stats as sp

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

embeddings = np.load('embeddings.npy')
sim_df = pd.read_csv('similarity_matrix.csv')
designers = sim_df['designer'].values

# For memory efficiency, compute in chunks rather than the full NxN matrix
CHUNK = 500
K = 50  # top-K nearest neighbours
cross_designer_scores = []

for start in tqdm(range(0, len(embeddings), CHUNK)):
    chunk = embeddings[start:start+CHUNK]
    sims = cosine_similarity(chunk, embeddings)  # (CHUNK x N)
    for i, row in enumerate(sims):
        global_i = start + i
        # Get top-K indices (excluding self)
        top_k = np.argsort(row)[::-1][1:K+1]
        neighbour_designers = designers[top_k]
        own_designer = designers[global_i]
        # How many distinct OTHER designers appear in the neighbourhood?
        other_designers = set(neighbour_designers) - {own_designer}
        cross_designer_scores.append(len(other_designers))

sim_df['cross_designer_diversity'] = cross_designer_scores
sim_df['persistence'] = sim_df['best_match'].map(persistence)
sim_df.to_csv('similarity_matrix_with_diversity26.csv', index=False)

# Now test: do persistent-trend images have higher cross-designer diversity?
from scipy import stats
p_div = sim_df[sim_df['persistence']=='persistent']['cross_designer_diversity']
s_div = sim_df[sim_df['persistence']=='spike_fade']['cross_designer_diversity']
t, p = stats.ttest_ind(p_div, s_div, equal_var=False)
print(f'Cross-designer diversity: persistent={p_div.mean():.2f} vs spike_fade={s_div.mean():.2f}, p={p:.4f}')


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
import umap

sim_df = pd.read_csv('similarity_matrix_with_diversity.csv')
embeddings = np.load('embeddings.npy')

COLOR_MAP = {
    'persistent':   '#0F6E6E',
    'spike_fade':   '#C0392B',
    'emerging':     '#B85C00',
    'peak_decline': '#6C3FC5',
    'faded':        '#888888',
}

# ── Figure 1: UMAP coloured by persistence label ─────────────────────
print('Running UMAP... (2–5 min)')
reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42, n_neighbors=15)
coords = reducer.fit_transform(embeddings)
sim_df['umap_x'] = coords[:,0]
sim_df['umap_y'] = coords[:,1]

fig, ax = plt.subplots(figsize=(11, 9))
for label, grp in sim_df.groupby('persistence'):
    ax.scatter(grp['umap_x'], grp['umap_y'],
               c=COLOR_MAP.get(label,'gray'), label=label, s=3, alpha=0.4)
ax.legend(title='Trend persistence', markerscale=5, fontsize=10)
ax.set_title('Figure 1 — CLIP embedding space (Vogue Runway, 5 seasons)', fontsize=13)
ax.set_xlabel('UMAP dimension 1'); ax.set_ylabel('UMAP dimension 2')
plt.tight_layout()
plt.savefig('fig1_umap_persistence.png', dpi=180, bbox_inches='tight')

print("prva dokoncana")

# ── Figure 2: Season evolution lines ─────────────────────────────────
import matplotlib.pyplot as plt

season_order = {
    'Spring 2024 Ready-to-Wear': 0,
    'Fall 2024 Ready-to-Wear':   1,
    'Spring 2025 Ready-to-Wear': 2,
    'Fall 2025 Ready-to-Wear':   3,
    'Spring 2026 Ready-to-Wear': 4,
    'Fall 2026 Ready-to-Wear':   5,
}
season_labels = ['SS24','AW24','SS25','AW25','SS26','AW26']
sim_df['season_idx'] = sim_df['show'].map(season_order)
keywords = [c for c in sim_df.columns if 'hot pants' in c or 'creamy yellow' in c
            or 'moccasins' in c or 'bubble hem' in c or 'wide leg trousers' in c
            or 'thong sandals' in c or 'Y2K fashion' in c or 'strapless dress' in c
            or 'barrel leg jeans' in c or 'ballet flat shoes' in c]
season_means = sim_df.groupby('season_idx')[keywords].mean()

fig, ax = plt.subplots(figsize=(12, 6))

for kw in keywords:
    label = sim_df[sim_df['best_match']==kw]['persistence'].mode()
    color = COLOR_MAP.get(label.iloc[0] if len(label) else 'faded', 'gray')
    
    # 1. Plot the line as usual (removed the label=kw property since we don't need a legend)
    ax.plot(season_means.index, season_means[kw], marker='o', color=color, linewidth=1.8)
    
    # 2. Get the position of the last data point (index 5 / AW26)
    x_pos = season_means.index[-1]
    y_pos = season_means[kw].iloc[-1]
    
    # 3. Add the text slightly to the right of the final point
    ax.text(
        x_pos + 0.08,             # Shift slightly right on the X-axis so it doesn't overlap the marker
        y_pos,                    # Align vertically with the last data point
        kw[:25],                  # The text label
        fontsize=8, 
        color=color,              # Match the text color to the line color
        va='center',              # Vertically center the text on the point
        ha='left'                 # Align text to the left of its anchor coordinate
    )

ax.set_xticks(range(6))
ax.set_xticklabels(season_labels)

# 4. Push the right plot border out a bit so the inline text labels don't get cut off
ax.set_xlim(-0.2, 5.8) 

ax.set_title('Mean CLIP similarity per keyword per season', fontsize=13)
ax.set_ylabel('Mean cosine similarity')

# Removed ax.legend() completely

plt.tight_layout()
plt.savefig('fig2_season_evolution.png', dpi=180, bbox_inches='tight')

print("druga dokoncana")

# ── Figure 3: Cross-designer diversity vs persistence ─────────────────
fig, ax = plt.subplots(figsize=(8, 5))
order = ['persistent','spike_fade','emerging','peak_decline','faded']
order = [o for o in order if o in sim_df['persistence'].unique()]
palette = {k:v for k,v in COLOR_MAP.items() if k in order}
sns.violinplot(data=sim_df, x='persistence', y='cross_designer_diversity',
               order=order, palette=palette, ax=ax, inner='quartile')
ax.set_title('Figure 3 — Cross-designer visual diversity by persistence', fontsize=13)
ax.set_xlabel('Trend persistence category')
ax.set_ylabel('# distinct designers in top-50 visual neighbours')
plt.tight_layout()
plt.savefig('fig3_cross_designer.png', dpi=180, bbox_inches='tight')

print("tretja dokoncana")

# ── Figure 4: Keyword similarity heatmap per season ───────────────────
fig, ax = plt.subplots(figsize=(13, 5))
season_means.columns = [k[:18] for k in season_means.columns]
season_means.index = season_labels
sns.heatmap(season_means.T, annot=True, fmt='.2f', cmap='RdPu',
            linewidths=0.4, ax=ax, cbar_kws={'label': 'cosine similarity'})
ax.set_title('Figure 4 — Keyword–season similarity heatmap', fontsize=13)
plt.tight_layout()
plt.savefig('fig4_heatmap.png', dpi=180, bbox_inches='tight')
print('All figures saved.')

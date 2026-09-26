import pandas as pd
import matplotlib.pyplot as plt

# 1. Your data
data = {
    'tau': [0.180, 0.185, 0.190, 0.195, 0.200, 0.205, 0.210, 0.215, 0.220, 0.225, 0.230, 0.235, 0.240, 0.245, 0.250, 0.255, 0.260, 0.265, 0.270, 0.275, 0.280],
    'mean_p': [8.832, 8.680, 8.346, 8.015, 7.465, 6.876, 6.045, 5.228, 4.413, 3.206, 2.229, 1.577, 1.126, 0.725, 0.463, 0.212, 0.071, 0.030, 0.012, 0.004, 0.001],
    'mean_f': [8.519, 8.227, 7.889, 7.442, 6.839, 6.169, 5.516, 4.812, 4.096, 3.429, 2.747, 2.191, 1.655, 1.075, 0.748, 0.494, 0.311, 0.146, 0.024, 0.008, 0.005],
    'cohens_d': [0.944, 0.954, 0.775, 0.891, 0.806, 0.799, 0.538, 0.395, 0.264, -0.232, -0.535, -0.587, -0.569, -0.401, -0.324, -0.440, -0.478, -0.447, -0.289, -0.175, -0.194],
    'p_perm': [0.001, 0.002, 0.014, 0.007, 0.009, 0.013, 0.102, 0.230, 0.429, 0.510, 0.082, 0.050, 0.057, 0.196, 0.377, 0.119, 0.039, 0.099, 0.407, 0.814, 0.962]
}
df = pd.DataFrame(data)

# 2. Initialize a single panel plot
fig, ax1 = plt.subplots(figsize=(10, 7))

# ── LEFT Y-AXIS: Activation Metrics & Effect Size ──
# Plot the Group Means (Measured from 0 to 9 keywords)
line1 = ax1.plot(df['tau'], df['mean_p'], color='#2E86C1', linewidth=2.5, marker='o', label='PERSISTENT Means (Left Axis)')
line2 = ax1.plot(df['tau'], df['mean_f'], color='#E67E22', linewidth=2.5, marker='s', label='LOW_INTEREST Means (Left Axis)')

# Plot Cohen's d on the same left axis
line3 = ax1.plot(df['tau'], df['cohens_d'], color='#4A2D8F', linewidth=2, linestyle='-.', marker='d', label="Cohen's d (Left Axis)")

ax1.set_ylabel("Activation Counts / Standardized Effect Size", fontsize=11)
ax1.set_xlabel("Similarity Threshold τ", fontsize=11)
ax1.axhline(0, color='black', linewidth=1, linestyle='-') # Base zero-line for Cohen's d
ax1.grid(True, alpha=0.3)

# Shading background regions based on threshold properties
ax1.fill_between(df['tau'], -2, 10, where=(df['tau'] <= 0.205), color='gray', alpha=0.05)
ax1.fill_between(df['tau'], -2, 10, where=(df['tau'] >= 0.225), color='red', alpha=0.04)

# ── RIGHT Y-AXIS: The Probability Scale (Luck Alarm) ──
ax2 = ax1.twinx()
line4 = ax2.plot(df['tau'], df['p_perm'], color='#C0392B', linewidth=2, linestyle='--', marker='x', label="p_perm (Right Axis)")
line5 = ax2.axhline(0.05, color='#C0392B', linestyle=':', linewidth=1.2, label='α = 0.05 Significance Line')

ax2.set_ylabel("Permutation p-value", color='#C0392B', fontsize=11)
ax2.tick_params(axis='y', labelcolor='#C0392B')
ax2.set_ylim(-0.05, 1.05) # Bound probability scale naturally from 0 to 1

# ── ANNOTATIONS & SELECTION ANCHOR ──
# The primary unbiased zero-crossing anchor at tau = 0.22
ax1.axvline(0.220, color='#0F6E6E', linestyle='--', linewidth=2)

# Dynamic labeling directly on the plot area
ax1.text(0.222, 1.5, 'τ = 0.220', color='#0F6E6E', fontweight='bold', fontsize=9)
ax1.text(0.232, 6.0, 'Raw Means Intersect\n(Direction Flips)', color='red', fontsize=9, fontweight='bold')

# Combine all generated plot components into a single clean legend block
all_lines = line1 + line2 + line3 + line4 + [line5]
all_labels = [l.get_label() for l in all_lines]
ax1.legend(all_lines, all_labels, loc='upper right', fontsize=9, framealpha=0.95)


plt.tight_layout()
plt.savefig('fig1_season_trajectory_edit.png', dpi=180, bbox_inches='tight')
plt.show()

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- 1. LOAD EMBEDDINGS AND METADATA ---
print("Loading data matrices...")
embeddings_raw = np.load('embeddings.npy')
embeddings_clean = np.load('embeddings_clean.npy')
meta = pd.read_csv('similarity_matrix.csv')

# Create a unique combined key for every show in the dataset
meta['show_id'] = meta['designer'] + " | " + meta['show']
unique_shows = meta['show_id'].dropna().unique()

print(f"Analyzing {len(unique_shows)} total runway collections across the entire dataset...")

# Storage lists for our metrics
raw_within_scores = []
clean_within_scores = []

raw_between_scores = []
clean_between_scores = []

# --- 2. GLOBAL MATRIX CALCULATIONS ---
# We loop through every show to calculate its internal vs external similarity profile
for current_show in tqdm(unique_shows, desc="Computing Global Similarity Baselines"):
    
    # Extract indices for the current show
    show_idx = meta[meta['show_id'] == current_show].index.values
    
    # Extract indices for all OTHER shows combined (to calculate "Between" metrics)
    other_idx = meta[meta['show_id'] != current_show].index.values
    
    # Skip calculations if a show has too few looks to compare
    if len(show_idx) < 2:
        continue
        
    # --- RAW IMAGES CALCULATIONS ---
    # Within-show similarity matrix
    sim_within_raw = cosine_similarity(embeddings_raw[show_idx])
    # Use triu_indices to ignore self-identity diagonal (similarity of image with itself)
    raw_within_scores.append(sim_within_raw[np.triu_indices_from(sim_within_raw, k=1)].mean())
    
    # Between-show similarity matrix (current show vs a sample of other shows to save memory)
    # Sampling 200 random outside images keeps computations fast and precise
    sampled_other_idx = np.random.choice(other_idx, size=min(200, len(other_idx)), replace=False)
    sim_between_raw = cosine_similarity(embeddings_raw[show_idx], embeddings_raw[sampled_other_idx])
    raw_between_scores.append(sim_between_raw.mean())
    
    # --- CLEAN IMAGES CALCULATIONS ---
    # Within-show similarity
    sim_within_clean = cosine_similarity(embeddings_clean[show_idx])
    clean_within_scores.append(sim_within_clean[np.triu_indices_from(sim_within_clean, k=1)].mean())
    
    # Between-show similarity
    sim_between_clean = cosine_similarity(embeddings_clean[show_idx], embeddings_clean[sampled_other_idx])
    clean_between_scores.append(sim_between_clean.mean())

# --- 3. DISPLAY GRAND METRICS SUMMARY ---
print("\n" + "="*50)
print("       GLOBAL BACKGROUND BIAS ANALYSIS SUMMARY       ")
print("="*50)

# Raw Averages
avg_raw_within = np.mean(raw_within_scores)
avg_raw_between = np.mean(raw_between_scores)
raw_delta = avg_raw_within - avg_raw_between

# Clean Averages
avg_clean_within = np.mean(clean_within_scores)
avg_clean_between = np.mean(clean_between_scores)
clean_delta = avg_clean_within - avg_clean_between

print(f"RAW IMAGES FOLDER (Backgrounds Intact):")
print(f"  └─ Global Within-Show Similarity:  {avg_raw_within:.4f}")
print(f"  └─ Global Between-Show Similarity: {avg_raw_between:.4f}")
print(f"  └─ Aesthetic Variance Delta:        {raw_delta:.4f}")

print(f"\nCLEAN IMAGES FOLDER (Backgrounds Stripped):")
print(f"  └─ Global Within-Show Similarity:  {avg_clean_within:.4f}")
print(f"  └─ Global Between-Show Similarity: {avg_clean_between:.4f}")
print(f"  └─ Aesthetic Variance Delta:        {clean_delta:.4f}")

print("-"*50)
global_bias_reduction = avg_raw_between - avg_clean_between
print(f"Global Background Noise Reduction: {global_bias_reduction:.4f}")
print("="*50)
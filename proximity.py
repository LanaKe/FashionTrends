import open_clip
import torch
import numpy as np
import pandas as pd

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

keywords = [
    'creamy yellow', 'hot pants', 'moccasins', 'bubble hem', 
    'wide leg trousers', 'Y2K fashion', 'thong sandals', 'strapless dress', 'barrel leg jeans',
    'ballet flat shoes'
]

# Encode trend keywords as text embeddings
with torch.no_grad():
    tokens = tokenizer(keywords)
    text_embs = model.encode_text(tokens)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    text_embs = text_embs.numpy()   # shape: (10, 512)

# Load pre-computed image embeddings (already L2-normalised)
img_embs = np.load('embeddings.npy')  # shape: (N, 512)

# Cosine similarity = dot product of L2-normalised vectors
# Result shape: (N_images, N_keywords)
sim_matrix = img_embs @ text_embs.T

valid_df = pd.read_csv('valid_images.csv')
sim_df = pd.DataFrame(sim_matrix, columns=keywords)

# Merge with image metadata (designer, season)
meta_df = pd.read_csv('all_scraped_vogue_images.csv')
meta_df = meta_df[meta_df['local_path'].isin(valid_df['local_path'])]
sim_df = pd.concat([meta_df[['designer','show']].reset_index(drop=True),
                    sim_df.reset_index(drop=True)], axis=1)

# Derived columns
sim_df['best_match']    = sim_df[keywords].idxmax(axis=1)
sim_df['best_score']    = sim_df[keywords].max(axis=1)
THRESHOLD = 0.18  # similarity score above which an image 'relates to' a keyword
sim_df['n_trends_activated'] = (sim_df[keywords] > THRESHOLD).sum(axis=1)

sim_df.to_csv('similarity_matrix_clean.csv', index=False)
print("threshold:", THRESHOLD)
print(sim_df[['best_match','best_score','n_trends_activated']].describe())

THRESHOLD = 0.22  # similarity score above which an image 'relates to' a keyword
sim_df['n_trends_activated'] = (sim_df[keywords] > THRESHOLD).sum(axis=1)

sim_df.to_csv('similarity_matrix22.csv', index=False)
print("threshold:", THRESHOLD)
print(sim_df[['best_match','best_score','n_trends_activated']].describe())

THRESHOLD = 0.26  # similarity score above which an image 'relates to' a keyword
sim_df['n_trends_activated'] = (sim_df[keywords] > THRESHOLD).sum(axis=1)

sim_df.to_csv('similarity_matrix.csv', index=False)
print("threshold:", THRESHOLD)
print(sim_df[['best_match','best_score','n_trends_activated']].describe())

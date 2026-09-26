import open_clip
import torch
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load model — downloads ~350MB on first run, cached after that
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f'Running on: {device}')

# Load validated image list
valid_df = pd.read_csv('valid_images_clean.csv')
image_paths = valid_df['local_path'].tolist()

# Embed in batches of 64 (adjust down to 32 if you get OOM errors on GPU)
BATCH_SIZE = 64
all_embeddings = []



with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_imgs = []
        valid_in_batch = []
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert('RGB'))
                batch_imgs.append(img)
                valid_in_batch.append(p)
            except Exception:
                pass  # skip any late-stage corrupt files
        if not batch_imgs:
            continue
        batch_tensor = torch.stack(batch_imgs).to(device)
        embs = model.encode_image(batch_tensor)
        # L2-normalise — standard for cosine similarity downstream
        embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embeddings.append(embs.cpu().numpy())

embeddings = np.vstack(all_embeddings)  # shape: (N, 512)
np.save('embeddings_clean.npy', embeddings)
print(f'Saved embeddings: {embeddings.shape}')
# e.g. Saved embeddings: (41106, 512)

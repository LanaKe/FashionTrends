from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('clean_runway_metadata.csv')
valid_paths = []
issues = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = row['local_path']
    if not path or not Path(path).exists():
        issues.append((path, 'missing file'))
        continue
    try:
        with Image.open(path) as img:
            w, h = img.size
            # Reject: too small for CLIP (need at least 100px)
            if w < 100 or h < 100:
                issues.append((path, f'too small: {w}x{h}'))
                continue
            # Reject: extreme landscape aspect ratio (likely not a runway look)
            # Runway looks are portrait (tall) — flag very wide images
            if w / h > 1.5:
                issues.append((path, f'landscape aspect: {w/h:.2f}'))
                continue
            valid_paths.append(path)
    except Exception as e:
        issues.append((path, f'corrupt: {e}'))

print(f'Valid: {len(valid_paths)} / Total: {len(df)}')
print(f'Issues: {len(issues)}')

# Save validated list
pd.DataFrame({'local_path': valid_paths}).to_csv('valid_images_clean.csv', index=False)
pd.DataFrame(issues, columns=['path','reason']).to_csv('rejected_images_clean.csv', index=False)

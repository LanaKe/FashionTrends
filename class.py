import torch
from PIL import Image
import open_clip

# 1. Load a CoCa (Contrastive Captioner) model instead of standard CLIP
model, _, transform = open_clip.create_model_and_transforms(
    'coca_ViT-B-32', pretrained='laion2b_s13b_b90k'
)
model.eval()

# 2. Preprocess your fashion image
image = transform(Image.open(r"runway_images/spring-2024-ready-to-wear/gucci/gucci_spring-2024-ready-to-wear_look003.jpg")).unsqueeze(0)

# 3. Generate actual text words using autoregressive decoding
with torch.no_grad(), torch.cuda.amp.autocast():
    generated_tokens = model.generate(image)

# 4. Decode the mathematical tokens back into strings
caption = open_clip.decode(generated_tokens[0]).split("<end_of_text>")[0].replace("<start_of_text>", "").strip()

print(f"Generated Label: {caption}")
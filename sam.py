import os
from pathlib import Path
from PIL import Image
from rembg import remove
from tqdm import tqdm
import csv

# --- 1. CONFIGURATION PATHS ---
input_root = Path(r"./images")
output_root = Path(r"./images_clean")
master_csv_path = Path(r"./clean_runway_metadata.csv")

# Find all original files
image_paths = list(input_root.glob("**/*.png"))
print(f"Found {len(image_paths)} runway images to process and catalog.")

# --- 2. INITIALIZE CSV FILE ---
# If the file doesn't exist yet, create it and write the master headers
if not master_csv_path.exists():
    with open(master_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['designer', 'show', 'look_number', 'local_path'])

# --- 3. PIPELINE LOOP ---
for img_path in tqdm(image_paths, desc="Processing & Cataloging Looks"):
    try:
        # Step A: Parse metadata directly out of the folder path structure
        designer_dir = img_path.parts[-3]   # e.g., 'kenzo'
        show_dir = img_path.parts[-2]       # e.g., 'spring-2024-ready-to-wear'
        filename = img_path.name            # e.g., 'kenzo-spring-2024-ready-to-wear-0.png'
        
        # Extract look number by grabbing the digits right before the .png extension
        look_number = filename.split('-')[-1].replace('.png', '')
        
        # Step B: Construct destination path
        relative_path = img_path.relative_to(input_root)
        target_output_path = output_root / relative_path
        
        # Step C: Background removal execution (if not already done)
        if not target_output_path.exists():
            target_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            input_img = Image.open(img_path)
            rgba_img = remove(input_img)
            
            # Composite onto solid white for optimal CLIP performance
            white_bg = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
            final_img = Image.alpha_composite(white_bg, rgba_img).convert("RGB")
            final_img.save(target_output_path)
        
        # Step D: Log this row directly to your new clean CSV spreadsheet
        with open(master_csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                designer_dir, 
                show_dir, 
                look_number, 
                str(target_output_path)  # Absolute Windows path to the background-removed photo
            ])
            
    except Exception as e:
        print(f"\nSkipping broken image or error on {img_path.name}. Error: {e}")

print(f"\n🎉 Process Complete!")
print(f"Clean Images Folder: {output_root}")
print(f"Clean Metadata Sheet: {master_csv_path}")
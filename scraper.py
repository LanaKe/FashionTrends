import vogue
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import csv
import os
import unidecode

#print(vogue.designer_to_shows('gucci'))
#print(vogue.designer_to_download_images('gucci', './images'))

designers = pd.read_csv('designers.csv')
shows = pd.read_csv('shows.csv')

'''for show in shows['show']:
    for name in designers['designer']:
        vogue.designer_show_to_csv(name, show, '.')
        #vogue.designer_show_to_download_images(name, show, './images') '''

master_csv_path = 'all_scraped_vogue_images.csv'
image_base_directory = Path(r"./images")
# Loop through every show and every designer
for show in shows['show']:
    for name in designers['designer']:
        
        # 1. Run the original function to get the data rows for this specific pair
        # (Pass save_path=None so it returns the data matrix to Python instead of saving a million tiny files)
        rows = vogue.designer_show_to_csv(name, show, save_path=None)
        
        # 2. If the function successfully found images, append them to the master file
        if rows:
            # Check if the master file exists yet. If not, we need to write headers first.
            file_exists = os.path.exists(master_csv_path)
            
            designer_clean = unidecode.unidecode(name.replace(' ', '-').replace('.', '-').replace('&', '').replace('+', '').replace('--', '-').lower())
            show_clean = unidecode.unidecode(show.replace(' ', '-').lower())
            
            updated_rows = []
            for i, row in enumerate(rows):
                designer_name = row[0]
                show_name = row[1]
                image_url = row[2]
                
                # 2. Replicated your exact downloaded file name format (.png)
                local_filename = f"{designer_clean}-{show_clean}-{i}.png"
                
                # 3. Use Path division ( / ) to auto-generate the perfect Windows absolute path
                full_local_path = image_base_directory / designer_clean / show_clean / local_filename
                
                # Append the row converting the path object into a clean string
                updated_rows.append([designer_name, show_name, image_url, str(full_local_path)])
            
            with open(master_csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['designer', 'show', 'image_url', 'local_path'])
                writer.writerows(updated_rows)

            print(f"✅ Appended {len(rows)} images for {name} - {show} to master file.")
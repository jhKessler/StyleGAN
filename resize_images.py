from PIL import Image
import os
from tqdm import tqdm

for img_file in tqdm(os.listdir("data/all-dogs")):
    image = Image.open(fr"data\all-dogs\{img_file}").resize((64, 64))
    image.save(f"fid_format_data/real/{img_file}")
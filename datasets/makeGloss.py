import os
import json
import uuid
from datetime import date
from tqdm import tqdm

VIDEO_DIR = "ISL_Gifs"
OUTPUT_JSON = "gloss_dataset.json"

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.gif', '.jpg'}

gloss_dataset = {}

use_uuid = False
counter = date.today().toordinal()

video_files = [
    f for f in os.listdir(VIDEO_DIR)
    if os.path.isfile(os.path.join(VIDEO_DIR, f)) and os.path.splitext(f)[1].lower() in VIDEO_EXTS
]

for filename in tqdm(video_files, desc="Processing videos"):
    file_path = os.path.join(VIDEO_DIR, filename)
    name, ext = os.path.splitext(filename)

    if use_uuid:
        video_id = str(uuid.uuid4())
    else:
        video_id = f"video_{counter:04d}"
        counter += 1

    gloss_sentence = name.replace("_", " ").strip()
    gloss_dataset[video_id] = gloss_sentence

    new_filename = f"{video_id}{ext}"
    new_path = os.path.join(VIDEO_DIR, new_filename)
    os.rename(file_path, new_path)

    print(f"Converted {filename} -> {new_filename}")

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(gloss_dataset, f, indent=4, ensure_ascii=False)

print(f"Gloss dataset saved to {OUTPUT_JSON}")

import json
from tqdm import tqdm
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

IN_PATH = r'D:\Work\text2sign\datasets\WLASL\WLASL_v0.3.json'
OUT_PATH = os.path.join(script_dir, 'WLASL_converted.json')

with open(IN_PATH, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

output_data = {}
for entry in tqdm(input_data):
    gloss = entry["gloss"]
    vid_ids = [f"{inst['video_id']}" for inst in entry["instances"]]
    output_data[gloss] = vid_ids
    
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2)
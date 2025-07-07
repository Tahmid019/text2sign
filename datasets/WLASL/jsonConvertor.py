import json
from tqdm import tqdm
import os
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

IN_PATH = r'D:\Work\text2sign\datasets\WLASL\WLASL_v0.3.json'
OUT_PATH = os.path.join(script_dir, 'WLASL_con2.json')

EXTERN_LINK = r'D:\Work\text2sign\datasets\WLASL\videos'

log_dir = r'datasets\WLASL\logs'
log_file = os.path.join(log_dir, "file_check.log")
os.makedirs(log_dir, exist_ok=True)


with open(IN_PATH, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

output_data = {}
for entry in tqdm(input_data):
    gloss = entry["gloss"]
    
    vid_ids = []
    for inst in entry["instances"]:    
        # vid_ids = [f"{inst['video_id']}" for inst in entry["instances"]]
        exist_path = os.path.join(EXTERN_LINK, inst['video_id'] + '.mp4')
        exist_status = os.path.isfile(exist_path)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] File '{exist_path}' exists: {exist_status}\n"
        
        with open(log_file, "a") as f:
            f.write(log_message)
        
        if exist_status:
            vid_ids.append(inst['video_id'])
    output_data[gloss] = vid_ids
    
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2)
import json
import os

DATASET_DIR = 'ISL_Gifs'
INP = 'inv_gloss.json'
exts = [".gif", ".mp4", ".webm", ".avi", ".jpg"]

def get_vid_path(text):
    with open(INP, 'r', encoding='utf-8') as f:
        gloss_data = json.load(f)
        
        
    vid_id = gloss_data[text]
    # Check for available file formats
    vid_path = None
    for ext in exts:
        candidate = os.path.join(DATASET_DIR, vid_id + ext)
        if os.path.isfile(candidate):
            vid_path = candidate
            break
    if vid_path is None:
        raise FileNotFoundError(f"No video file found for {vid_id} in supported formats: {exts}")
    
    return vid_path
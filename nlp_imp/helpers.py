import json
import os
import torch
from sentence_transformers import SentenceTransformer, util
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_DIR = 'nlp_imp/ISL_Gifs'
INP = 'nlp_imp/inv_gloss.json'

KEYDATASET = 'nlp_imp/keyframe_videos'
KEY_SUFF = 'output_'

logging.info(f'Loading json: {INP}')
with open(INP) as f:
    sentence2vid = json.load(f)

exts = [".gif", ".mp4", ".webm", ".avi", ".jpg"]
data = torch.load('nlp_imp/corpus_data.pt')
sentences = data['sentences']
corpus_embeddings = data['embeddings']

model = SentenceTransformer('all-MiniLM-L6-v2')

def find_video_id(user_query: str, threshold: float = 0.7):
    query_emb = model.encode(user_query, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(query_emb, corpus_embeddings)[0]
    
    top_idx = torch.argmax(cos_scores).item()
    top_score = cos_scores[top_idx].item()
    
    if top_score >= threshold:
        matched_sentence = sentences[top_idx]
        return sentence2vid[matched_sentence], top_score, matched_sentence
    else:
        return None, top_score, None



def get_vid_path(text): 
    vid_id, score, matched = find_video_id(text.lower())
    logging.info(f"Matched “{matched}” (score={score:.2f}) → {vid_id}")
    vid_path = None
    keyframe_path = None
    for ext in exts:
        candidate = os.path.join(DATASET_DIR, vid_id + ext)
        if ext is not '.jpg': keyframe_path = os.path.join(KEYDATASET, KEY_SUFF + vid_id + ".mp4")
        else: keyframe_path = None
        if os.path.isfile(candidate):
            vid_path = candidate
            break
    if vid_path is None:
        raise FileNotFoundError(f"No video file found for {vid_id} in supported formats: {exts}")
    
    return vid_path, score, matched, keyframe_path
from imports import *
from config import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HELPER_DATASET_DIR = f'{DATASET_DIR}/{ISL_DATASET_DIR}'
INP = f'{DATASET_DIR}/{INV_GLOSS}'
HELPER_KEYDATASET = f'{DATASET_DIR}/{KEYFRAME_DATASET_DIR}'
HELPER_KEYDATASET_HAND = f'{DATASET_DIR}/{KEYHAND_DATASET_DIR}'


logging.info(f'Loading json: {INP}')
with open(INP) as f:
    sentence2vid = json.load(f)

exts = T2SL_EXTS
data = torch.load(f'{PROJECT_ROOT}/{DATASET_DIR}/{CORPUS_DATASET}_2.pt',  map_location=torch.device('cpu'))
sentences = data['sentences']
corpus_embeddings = data['embeddings']

model = SentenceTransformer('all-mpnet-base-v2')

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
    keyframe_hand_path = None
    for ext in exts:
        candidate = os.path.join(HELPER_DATASET_DIR, vid_id + ext)
        if ext is not '.jpg': 
            keyframe_path = os.path.join(HELPER_KEYDATASET, KEY_SUFF + vid_id + ".mp4")
            keyframe_hand_path = os.path.join(HELPER_KEYDATASET_HAND, KEY_SUFF + vid_id + ".mp4")
        else: 
            keyframe_path = None
            keyframe_hand_path = None
            
        if os.path.isfile(candidate):
            vid_path = candidate
            break
        
    if vid_path is None:
        raise FileNotFoundError(f"No video file found for {vid_id} in supported formats: {exts}")
    
    return vid_path, score, matched, keyframe_path, keyframe_hand_path



def get_top_k_matches(user_inp: str, k: int = 5):
    inp_emb = model.encode(user_inp, convert_to_tensor=True)
    cos_scores = util.cos_sim(inp_emb, corpus_embeddings)[0]
    
    topk = torch.topk(cos_scores, k)
    res = []
    for score, idx in zip(topk.values, topk.indices):
        sent = sentences[idx]
        vid_id = sentence2vid.get(sent, None)
        res.append((sent, vid_id, score.item()))
    return res
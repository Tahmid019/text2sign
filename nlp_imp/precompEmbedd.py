import json
import logging
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INV_GLOSS = 'inv_gloss.json'

logging.info(f'Loading json: {INV_GLOSS}')
with open(INV_GLOSS) as f:
    sentence2vid = json.load(f)

logging.info('Loading Model')
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = list(sentence2vid.keys())
logging.info(f'Encoding {len(sentences)} sentences...')

corpus_embeddings = []
for sentence in tqdm(sentences, desc="Encoding sentences"):
    embedding = model.encode(sentence, convert_to_tensor=True)
    corpus_embeddings.append(embedding)

corpus_embeddings = torch.stack(corpus_embeddings)

output_path = 'corpus_data.pt'
torch.save({'sentences': sentences, 'embeddings': corpus_embeddings}, output_path)
logging.info(f'Saved embeddings and sentences to {output_path}')

# import json
# import logging
# import torch
# from sentence_transformers import SentenceTransformer, util

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class GlossRetriever:
#     def __init__(self):
#         self.model = None
#         self.sentences = None
#         self.embeddings = None 
#         self.sentence2vis = None
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         logging.info(f"Using device: {self.device}")
        
#     def load_resources(self):
        
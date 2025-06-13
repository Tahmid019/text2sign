# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Text2GlossTransformer(nn.Module):
    def __init__(self, text_vocab_size, gloss_vocab_size):
        super().__init__()
        self.text_embed = nn.Embedding(text_vocab_size, 256)
        self.gloss_embed = nn.Embedding(gloss_vocab_size, 256)
        self.transformer = nn.Transformer(
            d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3
        ).to(device)
        self.fc = nn.Linear(256, gloss_vocab_size)

    def forward(self, src, tgt):
        src = self.text_embed(src).permute(1,0,2) # S, B, E
        tgt = self.gloss_embed(tgt).permute(1,0,2)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(device)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.fc(output).permute(1,0,2) # B, S, V

class Gloss2Pose(nn.Module):
    def __init__(self, gloss_vocab_size, pose_dim=51): # 17 key points * 3 (x, y, conf)
        super().__init__()
        self.embed = nn.Embedding(gloss_vocab_size, 128)
        self.conv = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, pose_dim, 3, padding=1)
        )

    def forward(self, gloss_seq):
        x = self.embed(gloss_seq).permute(0,2,1) # B, C, S
        return self.conv(x).permute(0,2,1) # B, S, D

class TextGlossDataset3(Dataset): # takes from .pth
    def __init__(self, processed_path):
        
        data = torch.load(processed_path, map_location=torch.device("cpu"))
        self.text_vocab  = data["text_vocab"]
        self.gloss_vocab = data["gloss_vocab"]
        self.inv_gloss   = data["inv_gloss"]

        # Pre‐tokenized (N, max_seq_len)
        self.text_matrix  = data["text_matrix"]
        self.gloss_matrix = data["gloss_matrix"]

        assert self.text_matrix.size(0) == self.gloss_matrix.size(0), "Mismatch in example count"

    def __len__(self):
        return self.text_matrix.size(0)

    def __getitem__(self, idx):
        text_indices  = self.text_matrix[idx]
        gloss_indices = self.gloss_matrix[idx]
        return text_indices, gloss_indices

    def decode_gloss(self, indices):
        return " ".join(
            [self.inv_gloss.get(int(idx), "<unk>") for idx in indices if idx not in {0, 1, 2}]
        )
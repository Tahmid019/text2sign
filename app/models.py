# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Updated Gloss2Pose model
class Gloss2Pose(nn.Module):
    def __init__(self, gloss_vocab_size, pose_dim=99, max_frames=30):
        super().__init__()
        self.embed = nn.Embedding(gloss_vocab_size, 128)
        self.max_frames = max_frames
        self.conv = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, pose_dim, 3, padding=1)
        )
        # Frame prediction layer
        self.frame_predictor = nn.Linear(128, max_frames)
        
    def forward(self, gloss_seq):
        x = self.embed(gloss_seq)  # (B, S, E)
        x = x.mean(dim=1)  # Global average pooling (B, E)
        
        # Predict number of frames
        frame_logits = self.frame_predictor(x)
        num_frames = torch.argmax(frame_logits, dim=1) + 1  # (B,)
        
        # Generate pose sequence
        x = x.unsqueeze(-1).repeat(1, 1, self.max_frames)  # (B, E, max_frames)
        x = self.conv(x)  # (B, pose_dim, max_frames)
        x = x.permute(0, 2, 1)  # (B, max_frames, pose_dim)
        
        # Mask extra frames
        mask = torch.arange(self.max_frames, device=x.device)[None, :] < num_frames[:, None]
        return x * mask.unsqueeze(-1), num_frames

# Fixed Text2GlossTransformer
class Text2GlossTransformer(nn.Module):
    def forward(self, src, tgt):
        src = self.text_embed(src).permute(1, 0, 2)
        tgt = self.gloss_embed(tgt).permute(1, 0, 2)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(device)
        memory = self.transformer.encoder(src)
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.fc(output).permute(1, 0, 2)

class TextGlossDataset3(Dataset): # takes from .pth
    def __init__(self, processed_path):
        
        data = torch.load(processed_path, map_location=torch.device("cpu"))
        self.text_vocab  = data["text_vocab"]
        self.gloss_vocab = data["gloss_vocab"]
        self.inv_gloss   = data["inv_gloss"]

        # Pre‐tokenized (N, max_seq_len)
        self.text_matrix  = data["text_matrix"]
        self.gloss_matrix = data["gloss_matrix"]
        self.pose_matrix = data['pose_matrix']

        assert self.text_matrix.size(0) == self.gloss_matrix.size(0), "Mismatch in example count"

    def __len__(self):
        return self.text_matrix.size(0)

    def __getitem__(self, idx):
        return {
            'text': self.text_matrix[idx],
            'gloss': self.gloss_matrix[idx],
            'pose': self.pose_matrix[idx]
        }

    def decode_gloss(self, indices):
        return " ".join(
            [self.inv_gloss.get(int(idx), "<unk>") for idx in indices if idx not in {0, 1, 2}]
        )
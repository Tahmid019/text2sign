import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# ---------- DATASET ----------
class LandmarkGlossDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.samples = []  # list of (landmarks_array, label_index)
        glosses = sorted(data.keys())
        self.label_map = {g: i for i, g in enumerate(glosses)}
        for gloss, entries in data.items():
            label = self.label_map[gloss]
            for e in entries:
                # each e has 'vid_id' and 'landmarks': list of [T, D]
                landmarks = np.array(e['landmarks'], dtype=np.float32)
                self.samples.append((landmarks, label))
        self.num_classes = len(glosses)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        landmarks, label = self.samples[idx]
        return torch.from_numpy(landmarks), torch.tensor(label, dtype=torch.long)

# ---------- MODEL (BiLSTM baseline) ----------
class BiLSTMSignModel(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.fc_in = nn.Linear(feature_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size*2),
            nn.Linear(hidden_size*2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: [B, T, D]
        x = self.fc_in(x)      # [B, T, H]
        x = self.dropout(x)
        outputs, (hn, _) = self.lstm(x)
        # hn: [layers*2, B, H]
        forward_last = hn[-2]
        backward_last = hn[-1]
        h = torch.cat([forward_last, backward_last], dim=1)  # [B, H*2]
        return self.classifier(h)

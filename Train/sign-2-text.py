import os
from pathlib import Path
import json
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from mediapipe import solutions
from tqdm import tqdm

# from config import *
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent   # Text2Sign/t2slt
# ---------- CONFIGURATION ----------
CONFIG = {
    "DATA_DIR": str(REPO_ROOT / "nlp_imp" / "ISL_Gifs"), 
    "GLOSS_MAP": str(REPO_ROOT / "nlp_imp" / "invGlossList.json"),  # JSON mapping: {"word": ["sample1", ...]}
    "BATCH_SIZE": 16,
    "EPOCHS": 50,
    "MAX_SEQ_LEN": 100,
    "FEATURE_DIM": 63,  # 21 landmarks x 3 coords
    "NUM_CLASSES": None,
    "LR": 1e-4,
    "CHECKPOINT_DIR": "./Train/checkpoints/",
    "LOG_DIR": "./Train/logs/",
    "exTS": ['.jpg', '.gif'],
}

print("DATA_DIR:", CONFIG["DATA_DIR"])
print("Exists:", Path(CONFIG["DATA_DIR"]).exists())
print("Contents:", list(Path(CONFIG["DATA_DIR"]).iterdir())[:5])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- HAND LANDMARK PROCESSOR ----------
hands_processor = solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- DATASET ----------
class SignDataset(Dataset):
    def __init__(self, data_dir, gloss_map, gloss_inv, max_seq_len, feature_dim):
        self.data_dir = data_dir
        self.samples = []
        self.gloss_map = gloss_map
        self.gloss_inv = gloss_inv
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        # build index: gloss_map maps word to list of sample IDs
        for gloss, vids in gloss_map.items():
            for vid in vids:
                found = False
                for ext in CONFIG['exTS']:
                    # if ext == '.jpg':
                    #     continue
                    candidate = Path(self.data_dir) / f"{vid}{ext}"
                    if candidate.exists():
                        self.samples.append((str(candidate), gloss_inv[gloss]))
                        found = True
                        break
                if not found:
                    print(f"[WARNING] No file found for ID={vid} in {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def _extract_landmarks(self, frames):
        coords = []
        for frame in frames:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_processor.process(img)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                arr = []
                for p in lm.landmark:
                    arr.extend([p.x, p.y, p.z])
                coords.append(arr)
            else:
                coords.append([0.0]*self.feature_dim)
        return np.array(coords, dtype=np.float32)

    def _load_frames(self, path):
        path = path.strip()
        # if GIF
        if os.path.isfile(path) and path.lower().endswith(('.gif', '.mp4', '.avi')):
            vid = imageio.get_reader(path)
            frames = [cv2.resize(frame, (224, 224)) for frame in vid]
        # if directory of images
        elif os.path.isdir(path):
            imgs = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg','.png'))])
            frames = [cv2.resize(cv2.imread(os.path.join(path,f)), (224,224)) for f in imgs]
        else:
            raise FileNotFoundError(path)
        return frames

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._load_frames(path)
        coords = self._extract_landmarks(frames)
        # pad/trim sequence
        if coords.shape[0] >= self.max_seq_len:
            coords = coords[:self.max_seq_len]
        else:
            pad = np.zeros((self.max_seq_len - coords.shape[0], self.feature_dim), dtype=np.float32)
            coords = np.vstack([coords, pad])
        return torch.from_numpy(coords), torch.tensor(label, dtype=torch.long)

# ---------- MODEL ----------
class TransformerSignModel(nn.Module):
    def __init__(
        self,
        seq_len,
        feature_dim,
        num_classes,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.3,
        frame_drop_prob=0.1
    ):
        super().__init__()
        # Input projection
        self.fc_in = nn.Linear(feature_dim, d_model)
        # Learned positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))  # +1 for cls token
        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Single encoder layer template
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False  #  permute manually
        )
        # Transformer encoder stack
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Adaptive frame dropout
        self.frame_dropout = nn.Dropout(frame_drop_prob)
        # Classification head: two-layer MLP with LayerNorm
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: [B, T, F]
        B, T, _ = x.size()
        # project features
        x = self.fc_in(x)  # [B, T, D]
        # prepend cls token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)  # [B, T+1, D]
        # add positional embeddings
        x = x + self.pos_emb[:, : T + 1]
        # (avoid dropping cls token)
        x[:, 1:] = self.frame_dropout(x[:, 1:])
        # transformer expects [S, B, D]
        x = x.permute(1, 0, 2)  # [T+1, B, D]
        x = self.transformer(x)  # [T+1, B, D]
        x = x.permute(1, 0, 2)  # [B, T+1, D]
        cls_out = x[:, 0, :]  # [B, D]
        # classification head
        return self.classifier(cls_out)

class LSTMSignModel(nn.Module):
    def __init__(self, seq_len, feature_dim, num_classes, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        # Bi‑LSTM encoder
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        # Pool the final hidden states from both directions
        self.fc = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x, lengths=None):
        # x: [B, T, F]
        # (optional) you can pack_padded_sequence here if you have per‐sample lengths
        outputs, (hn, cn) = self.lstm(x)
        # hn: [num_layers*2, B, hidden]
        # take last layer's forward and backward hidden states
        last_forward = hn[-2]   # [B, hidden]
        last_backward = hn[-1]  # [B, hidden]
        h = torch.cat([last_forward, last_backward], dim=1)  # [B, hidden*2]
        return self.fc(h)



# ---------- TRAINING LOOP ----------
def train_epoch(model, loader, criterion, optimizer, epoch, writer):
    model.train()
    total_loss = total_corr = total_samples = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(data)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        total_corr += (preds.argmax(1) == labels).sum().item()
        total_samples += data.size(0)
    writer.add_scalar('Loss/train', total_loss/total_samples, epoch)
    writer.add_scalar('Acc/train', total_corr/total_samples, epoch)
    print(f"Epoch {epoch}: loss {total_loss/total_samples:.4f}, acc {total_corr/total_samples:.4f}")
    return total_loss/total_samples, total_corr/total_samples

# ---------- MAIN ----------
def main():
    # load gloss map
    with open(CONFIG['GLOSS_MAP']) as f:
        gloss_map = json.load(f)
    labels = sorted(gloss_map.keys())
    gloss_inv = {g:i for i,g in enumerate(labels)}
    CONFIG['NUM_CLASSES'] = len(labels)

    dataset = SignDataset(CONFIG['DATA_DIR'], gloss_map, gloss_inv,
                          CONFIG['MAX_SEQ_LEN'], CONFIG['FEATURE_DIM'])
    loader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True,
                        num_workers=4, pin_memory=True)

    # model = TransformerSignModel(CONFIG['MAX_SEQ_LEN'], CONFIG['FEATURE_DIM'], CONFIG['NUM_CLASSES']).to(device)
    model = LSTMSignModel(
        seq_len    = CONFIG['MAX_SEQ_LEN'],
        feature_dim= CONFIG['FEATURE_DIM'],
        num_classes= CONFIG['NUM_CLASSES']
    ).to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LR'])

    os.makedirs(CONFIG['CHECKPOINT_DIR'], exist_ok=True)
    writer = SummaryWriter(CONFIG['LOG_DIR'])
    losses = []
    accs = []
    for epoch in tqdm(range(1, CONFIG['EPOCHS']+1)):
        loss, acc = train_epoch(model, loader, criterion, optimizer, epoch, writer)
        losses.append(loss)
        accs.append(acc)
        torch.save(model.state_dict(), os.path.join(CONFIG['CHECKPOINT_DIR'], f"epoch{epoch:02d}.pt"))
    writer.close()
    print("Training complete.")
    df = pd.DataFrame({
        'Epochs' : range(len(losses)),
        'Losses' : losses,
        'Accuracies' : accs
    })
    df.to_csv('Train/Loss_Acc.csv', index=False)

if __name__ == '__main__':
    main()

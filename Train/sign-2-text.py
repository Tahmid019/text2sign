import os
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

# ---------- CONFIGURATION ----------
CONFIG = {
    "DATA_DIR": "./nlp_imp/ISL_Gifs", 
    "GLOSS_MAP": "./nlp_imp/gloss_dataset.json",  # JSON mapping: {"word": ["sample1", ...]}
    "BATCH_SIZE": 16,
    "EPOCHS": 50,
    "MAX_SEQ_LEN": 100,
    "FEATURE_DIM": 63,  # 21 landmarks x 3 coords
    "NUM_CLASSES": None,
    "LR": 1e-4,
    "CHECKPOINT_DIR": "./Train/checkpoints/",
    "LOG_DIR": "./Train/logs/"
}

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
        self.gloss_inv = gloss_inv
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        # build index: gloss_map maps word to list of sample IDs
        for gloss, ids in gloss_map.items():
            for sid in ids:
                path = os.path.join(data_dir, sid)
                if os.path.exists(path):
                    self.samples.append((path, gloss_inv[gloss]))

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
    def __init__(self, seq_len, feature_dim, num_classes, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.fc_in = nn.Linear(feature_dim, d_model)
        pe = self._positional_encoding(seq_len, d_model)
        self.register_buffer('pos_enc', pe)
        encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    def _positional_encoding(self, seq_len, d_model):
        pos = torch.arange(seq_len).unsqueeze(1)
        i = torch.arange(d_model).unsqueeze(0)
        angle = pos / (10000 ** (2 * (i//2) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        return pe.unsqueeze(1)

    def forward(self, x):
        # x: [B, T, F]
        x = self.fc_in(x) + self.pos_enc.permute(1,0,2)
        x = x.permute(1,0,2)  # [T,B,D]
        x = self.transformer(x)
        x = x.permute(1,2,0)  # [B,D,T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

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

    model = TransformerSignModel(CONFIG['MAX_SEQ_LEN'], CONFIG['FEATURE_DIM'], CONFIG['NUM_CLASSES']).to(device)
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

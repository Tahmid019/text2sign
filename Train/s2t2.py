import os
import json
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path

# ---------- CONFIGURATION ----------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LANDMARKS_JSON = REPO_ROOT / 'datasets' / 'augmented_gloss_landmarks.json'
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
LOG_DIR = SCRIPT_DIR / 'logs'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# ---------- TRAINING LOOP ----------

def train_epoch(model, loader, criterion, optimizer, epoch, writer):
    model.train()
    total_loss = total_correct = total = 0
    for data, labels in loader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += data.size(0)
    avg_loss = total_loss / total
    avg_acc = total_correct / total
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Accuracy', avg_acc, epoch)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    return avg_loss, avg_acc

# ---------- MAIN ----------

def main():
    # Prepare dataset
    dataset = LandmarkGlossDataset(LANDMARKS_JSON)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Model, loss, optimizer
    feature_dim = dataset.samples[0][0].shape[1]
    model = BiLSTMSignModel(feature_dim, dataset.num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Prepare logging & checkpoints
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    best_acc = 0
    losses = []
    accs = []
    for epoch in range(1, EPOCHS+1):
        loss, acc = train_epoch(model, loader, criterion, optimizer, epoch, writer)
        # save best
        losses.append(loss)
        accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / 'best_model.pt')
    df = pd.DataFrame({
        'Losses' : losses,
        'Accuracies' : accs,
    })
    df.to_csv('Train3.csv')
    writer.close()
    print(f"Training complete. Best Acc: {best_acc:.4f}")

if __name__ == '__main__':
    main()

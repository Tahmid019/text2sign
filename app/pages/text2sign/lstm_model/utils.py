import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import cv2

class LSTM_t2s(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTM_t2s, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim * 30)
        self.output_dim = output_dim

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        out = self.fc(hn[-1])
        return out.view(-1, 30, self.output_dim)
    
def tokenize(text, vocab, max_len=10):
    tokens = text.lower().split()
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return torch.tensor(ids[:max_len]).unsqueeze(0)

def render_keypoints(frame, keypoints):
    keypoints = keypoints.reshape(-1, 3)  
    for x, y, v in keypoints[:50]:  
        if v > 0:
            cv2.circle(frame, (int(x * 640), int(y * 480)), 3, (0, 255, 0), -1)
    return frame
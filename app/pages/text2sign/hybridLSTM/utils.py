import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)   # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)       
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))       # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class Text2SignHybrid(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 trans_dim: int,
                 lstm_hidden: int,
                 num_layers: int,
                 output_dim: int,
                 seq_len: int = 30,
                 num_heads: int = 4,
                 tcn_channels: list = [128, 64]):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model=embed_dim, max_len=seq_len)
        
        # ----- Transformer Encoder --------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead= num_heads,
            dim_feedforward=trans_dim,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # --------- LSTM Decoder ------------
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # -------------- Temporal Convolutional Refinement ---------------
        tcn_layers = []
        in_ch = lstm_hidden
        for out_ch in tcn_channels:
            tcn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch)
            ]
            in_ch = out_ch
        self.tcn = nn.Sequential(*tcn_layers)
        
        # ---- Final projection ---
        self.fc = nn.Linear(tcn_channels[-1], output_dim)
        

    def forward(self, x):
        B = x.size(0)
        
        emb = self.embedding(x)                # (B, T_text, embed_dim)
        emb = self.pos_enc(emb)                # "
        
        tr_in = emb.permute(1,0,2)             # (T_text, B, D)
        tr_out = self.transformer(tr_in)       # (T_text, B, D)
        tr_out = tr_out.permute(1,0,2)         # (B, T_text, D)
        

        token_ctx = tr_out[:, -1, :].unsqueeze(1)  # (B,1,D)
        dec_input = token_ctx.repeat(1, self.seq_len, 1)  # (B, seq_len, D)
        
        lstm_out, _ = self.lstm(dec_input)     # (B, seq_len, lstm_hidden)
        
        #  (B, C, T)
        tcn_in = lstm_out.permute(0,2,1)      # (B, lstm_hidden, seq_len)
        tcn_out = self.tcn(tcn_in)             # (B, last_ch, seq_len)
        
        tcn_out = tcn_out.permute(0,2,1)       # (B, seq_len, last_ch)
        out   = self.fc(tcn_out)             # (B, seq_len, output_dim)
        return out

    
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
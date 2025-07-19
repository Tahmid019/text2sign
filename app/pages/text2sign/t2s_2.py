import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from mediapipe.python.solutions.holistic import POSE_CONNECTIONS, HAND_CONNECTIONS

# from lstm_model.utils import LSTM_t2s, tokenize, render_keypoints

# class LSTM_t2s(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
#         super(LSTM_t2s, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim * 30)
#         self.output_dim = output_dim

#     def forward(self, x):
#         embedded = self.embedding(x)
#         _, (hn, _) = self.lstm(embedded)
#         out = self.fc(hn[-1])
#         return out.view(-1, 30, self.output_dim)

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


# =====================================================================================================

def tokenize(text, vocab, max_len=10):
    tokens = text.lower().split()
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return torch.tensor(ids[:max_len]).unsqueeze(0)


SKELETON_CONNECTIONS = set(POSE_CONNECTIONS) | set(HAND_CONNECTIONS)

N_pose = 33
N_hand = 21

def render_keypoints(frame, keypoints):
    h, w, _ = frame.shape
    pts = keypoints.reshape(-1, 3)        

    pose_pts = pts[:N_pose]
    left_pts = pts[N_pose   :N_pose+N_hand]
    right_pts= pts[N_pose+N_hand : N_pose+2*N_hand]

    def draw_part(pts, connections, color_line, color_dot):
        for i,j in connections:
            x1,y1,v1 = pts[i]
            x2,y2,v2 = pts[j]
            if v1>0 and v2>0:
                p1 = (int(x1*w), int(y1*h))
                p2 = (int(x2*w), int(y2*h))
                cv2.line(frame, p1, p2, color_line, 2)
        for x,y,v in pts:
            if v>0:
                cv2.circle(frame, (int(x*w), int(y*h)), 3, color_dot, -1)

    draw_part(pose_pts, POSE_CONNECTIONS, (0,255,0), (0,128,0))
    draw_part(left_pts, HAND_CONNECTIONS, (255,0,0), (128,0,0))
    draw_part(right_pts, HAND_CONNECTIONS, (0,0,255), (0,0,128))

    return frame



MODEL_PATH = "app/models/t2s_models"

@st.cache_resource
def load_models():
    models = {}
    vocab = None
    for file in os.listdir(MODEL_PATH):
        if file.endswith(".pth"):
            model_name = file[:-4]
            vocab_path = os.path.join(MODEL_PATH, model_name + "_vocab.npy")
            vocab_dict = np.load(vocab_path, allow_pickle=True).item()
            # model = LSTM_t2s(vocab_size=len(vocab_dict), embed_dim=128, hidden_dim=512, output_dim=1662)
            model = Text2SignHybrid(
                vocab_size  = len(vocab_dict),
                output_dim  = 1662,        
                seq_len     = 30,          
                embed_dim   = 128,        
                trans_dim   = 256,         
                lstm_hidden = 512,         
                num_layers  = 2,           
                num_heads   = 4,           
                tcn_channels= [128, 64]
            )

            model.load_state_dict(torch.load(os.path.join(MODEL_PATH, file), map_location="cpu"))
            model.eval()
            models[model_name] = (model, vocab_dict)
    return models
    
def t2s_2():
    st.title("Text-to-Sign Language Keypoint Generator")

    models_dict = load_models()
    model_names = list(models_dict.keys())

    selected_model_name = st.selectbox("Select a model", model_names, index=0)
    model, vocab = models_dict[selected_model_name]

    text_input = st.text_input("Enter gloss/text", "are you hungry")

    if st.button("Generate"):
        input_tensor = tokenize(text_input, vocab).long()
        with torch.no_grad():
            pred_keypoints = model(input_tensor).squeeze(0).numpy()  # (30, 1662)

        st.session_state.pred_keypoints = pred_keypoints
        st.session_state.current_text = text_input

    if "pred_keypoints" in st.session_state:
        st.subheader(f"Generated frames for: {st.session_state.current_text}")
        frame_idx = st.slider("Frame", 0, 29, 0)

        frame = np.ones((480, 640, 3), dtype=np.uint8)
        keypoints = st.session_state.pred_keypoints[frame_idx]
        rendered = render_keypoints(frame, keypoints)
        st.image(rendered, channels="BGR", caption=f"Frame {frame_idx + 1}")
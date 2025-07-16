import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import cv2

# from lstm_model.utils import LSTM_t2s, tokenize, render_keypoints

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
            model = LSTM_t2s(vocab_size=len(vocab_dict), embed_dim=128, hidden_dim=512, output_dim=1662)
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
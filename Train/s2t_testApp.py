import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
import mediapipe as mp
from s2t2 import BiLSTMSignModel  
from pathlib import Path
import json

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path('Train/checkpoints/best_model.pt')
LABELS_JSON = Path('datasets/augmented_gloss_landmarks.json')
SEQ_LEN = 60

# Load label map
def load_labels():
    with open(LABELS_JSON, 'r') as f:
        labels = sorted(json.load(f).keys())
    return labels

# Load model
def load_model():
    dummy_data = torch.randn(1, SEQ_LEN, 126).to(DEVICE) 
    model = BiLSTMSignModel(225, len(labels)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# Landmark extractor
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(frame_rgb)
    points = []
    #pose: 33 landmarks
    if result.pose_landmarks:
        for lm in result.pose_landmarks.landmark:
            points.extend([lm.x, lm.y, lm.z])
    else:
        points.extend([0.0]*33*3)
    
    # Left hand: 21 landmarks
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark:
            points.extend([lm.x, lm.y, lm.z])
    else:
        points.extend([0.0] * 21 * 3)

    # Right hand: 21 landmarks
    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark:
            points.extend([lm.x, lm.y, lm.z])
    else:
        points.extend([0.0] * 21 * 3)

    return points  # 75*3 = 225

# Streamlit UI
st.title("Real-time Sign to Text")
frame_placeholder = st.empty()
text_placeholder = st.markdown("**Prediction:** _Waiting..._")

# Load model and labels
labels = load_labels()
model = load_model()

cap = cv2.VideoCapture(0)
data_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    landmarks = extract_landmarks(frame)
    data_buffer.append(landmarks)
    if len(data_buffer) > SEQ_LEN:
        data_buffer.pop(0)

    if len(data_buffer) == SEQ_LEN:
        with torch.no_grad():
            inp = torch.tensor([data_buffer], dtype=torch.float32).to(DEVICE)
            pred = model(inp).argmax(dim=1).item()
            text_placeholder.markdown(f"**Prediction:** `{labels[pred]}`")

    frame_placeholder.image(frame, channels="BGR")

cap.release()

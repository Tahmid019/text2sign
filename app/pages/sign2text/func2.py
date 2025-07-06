import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import json
from keras.models import load_model

# local imports
from .cnn1dLstm import build_cnn_lstm_attention, Attention

# --- Config ---
SEQUENCE_LEN = 20
FEATURE_DIM = 126
MODEL_WEIGHTS = "app/models/best_model_CNN1s-LSTM.h5"
LABELS_PATH = "datasets/WLASL/WLASL_con_test.json"

# --- Load model and labels ---
@st.cache_resource
def load_labels_and_model():
    # Load label list
    with open(LABELS_PATH, 'r') as f:
        labels = sorted(json.load(f).keys())

    # Option A: Load full model with custom Attention layer
    model = load_model(
        MODEL_WEIGHTS,
        custom_objects={'Attention': Attention}
    )

    # Option B: Rebuild architecture and load weights only
    # model = build_cnn_lstm_attention(SEQUENCE_LEN, FEATURE_DIM, len(labels))
    # model.load_weights(MODEL_WEIGHTS)

    return labels, model

labels, model = load_labels_and_model()

# --- Mediapipe landmark extractor ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

def extract_landmarks_126(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(frame_rgb)

    points = []
    def flatten_landmarks(landmarks, count=21):
        if landmarks:
            for lm in landmarks.landmark:
                points.extend([lm.x, lm.y])
        else:
            points.extend([0.0] * count * 2)

    # Pose: nose, shoulders, elbows, wrists
    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        key_idxs = [0, 11, 12, 13, 14, 15, 16]
        for i in key_idxs:
            points.extend([lm[i].x, lm[i].y])
    else:
        points.extend([0.0] * 7 * 2)

    # Hands
    flatten_landmarks(result.left_hand_landmarks)
    flatten_landmarks(result.right_hand_landmarks)

    return points if len(points) == FEATURE_DIM else [0.0] * FEATURE_DIM

# --- Streamlit app ---

def s2t():
    st.title("Real-time Sign Language Recognition")
    st.info("Make sure your hands and upper body are visible in the webcam.")

    frame_placeholder = st.empty()
    text_placeholder = st.markdown("**Prediction:** _Waiting..._")

    cap = cv2.VideoCapture(0)
    data_buffer = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            landmarks = extract_landmarks_126(frame)
            data_buffer.append(landmarks)
            if len(data_buffer) > SEQUENCE_LEN:
                data_buffer.pop(0)

            if len(data_buffer) == SEQUENCE_LEN:
                seq = np.expand_dims(np.array(data_buffer), axis=0)
                preds = model.predict(seq)[0]
                idx = np.argmax(preds)
                conf = preds[idx]
                text_placeholder.markdown(f"**Prediction:** `{labels[idx]}` (conf: {conf:.2f})")

            frame_placeholder.image(frame, channels="BGR")

    finally:
        cap.release()
        holistic.close()


if __name__ == "__main__":
    s2t()

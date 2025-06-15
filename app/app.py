import streamlit as st
import torch
import numpy as np
import cv2
import tempfile
from models import Text2GlossTransformer, Gloss2Pose, TextGlossDataset3
from pathlib import Path
import os
import re

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

POSE_CONNECTIONS = [
    # Face outline
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # Body
    (9, 10),  # Mid hips to mid shoulders
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Arms
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32)
]

current_dir = Path(__file__).parent

@st.cache_resource
def load_models():
    dataset = TextGlossDataset3(current_dir / 'wlasl_dataset.pth')
    
    t2g_model = Text2GlossTransformer(
        len(dataset.text_vocab),
        len(dataset.gloss_vocab)
    ).to(DEVICE)
    t2g_model.load_state_dict(
        torch.load(current_dir / 't2g_model_weights.pth', map_location=DEVICE),
        strict=False
    )
    t2g_model.eval()
    
    g2p_model = Gloss2Pose(len(dataset.gloss_vocab)).to(DEVICE)
    
    g2p_state = torch.load(current_dir / 'g2p_model_weights.pth', map_location=DEVICE)
    embed_key = 'embed.weight'

    if embed_key in g2p_state:
        loaded_weight = g2p_state[embed_key]
        current_vocab_size = g2p_model.embed.num_embeddings
        loaded_vocab_size = loaded_weight.size(0)
        
        if current_vocab_size > loaded_vocab_size:
            pad_size = current_vocab_size - loaded_vocab_size
            pad_tensor = torch.zeros(pad_size, loaded_weight.size(1), device=loaded_weight.device)
            g2p_state[embed_key] = torch.cat([loaded_weight, pad_tensor], dim=0)
        elif current_vocab_size < loaded_vocab_size:
            g2p_state[embed_key] = loaded_weight[:current_vocab_size]

    g2p_model.load_state_dict(g2p_state, strict=False)
    g2p_model.eval()
    
    return t2g_model, g2p_model, dataset


def render_pose_frame(pose, frame_size=(512, 512)):
    """Render a single pose frame with proper normalization"""
    frame = np.zeros((*frame_size, 3), dtype=np.uint8)
    keypoints = pose.reshape(-1, 3)
    
    # Normalize keypoints to [0,1] range
    min_val, max_val = keypoints[:, :2].min(), keypoints[:, :2].max()
    if max_val - min_val > 1e-8:
        keypoints[:, :2] = (keypoints[:, :2] - min_val) / (max_val - min_val)
    
    # Scale to frame size with padding
    padding = 50
    keypoints[:, 0] = keypoints[:, 0] * (frame_size[1] - 2*padding) + padding
    keypoints[:, 1] = keypoints[:, 1] * (frame_size[0] - 2*padding) + padding
    
    # Draw connections
    for i, j in POSE_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            if keypoints[i, 2] > 0.2 and keypoints[j, 2] > 0.2:
                cv2.line(
                    frame,
                    (int(keypoints[i, 0]), int(keypoints[i, 1])),
                    (int(keypoints[j, 0]), int(keypoints[j, 1])),
                    (255, 166, 2), 3
                )
    
    # Draw points
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.2:
            color = (0, 255, 255)  # Yellow for most points
            # Color code important joints
            if i in [15, 16]:  # Hands
                color = (0, 165, 255)  # Orange
            elif i in [0, 1, 2, 3, 4]:  # Face
                color = (255, 0, 0)  # Blue
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
    
    return frame

def preprocess_text(text):
    """
    Preprocess input text to match how the training data was processed
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters (keep basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9.,!?']", " ", text)
    
    # Handle contractions (optional)
    contraction_map = {"don't": "do not", "can't": "cannot"}
    for k, v in contraction_map.items():
        text = text.replace(k, v)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Add sentence boundaries (if your model expects them)
    if not text.startswith("<sos>"):
        text = "<sos> " + text
    if not text.endswith("<eos>"):
        text = text + " <eos>"
    
    return text

def text_to_sign(text, t2g_model, g2p_model, dataset):
    """Convert text to sign language video with proper pose normalization"""
    # Tokenize input text
    tokens = [dataset.text_vocab.get(w.lower(), dataset.text_vocab["<unk>"]) 
              for w in text.split()]
    tokens = [dataset.text_vocab["<sos>"]] + tokens + [dataset.text_vocab["<eos>"]]
    src_tokens = torch.tensor([tokens]).to(DEVICE)
    
    print("tokens: ", tokens)
    
    # Generate gloss sequence
    gloss_seq = [dataset.gloss_vocab["<sos>"]]
    for _ in range(50):  # Max output length
        decoder_input = torch.tensor(gloss_seq, device=DEVICE).unsqueeze(0)

        # print(decoder_input)
        with torch.no_grad():
            logits = t2g_model(src_tokens, decoder_input)
        next_id = logits[0, -1].argmax().item()
        print("next id: ", next_id)
        if next_id == dataset.gloss_vocab["<eos>"]:
            break
        gloss_seq.append(next_id)
    
    # Convert gloss IDs to text
    gloss_text = ' '.join([dataset.inv_gloss.get(idx, '<unk>') 
                          for idx in gloss_seq[1:-1]])  # Remove <sos> and <eos>
    
    # Generate poses
    gloss_tensor = torch.tensor([gloss_seq[1:]]).to(DEVICE)
    with torch.no_grad():
        poses = g2p_model(gloss_tensor).cpu().numpy()[0]
    
    # Normalize and center poses
    for i in range(len(poses)):
        # Get keypoints for this frame
        frame_keypoints = poses[i].reshape(-1, 3)
        
        # Normalize to [0,1] range
        min_val = frame_keypoints[:, :2].min()
        max_val = frame_keypoints[:, :2].max()
        if max_val - min_val > 1e-8:
            frame_keypoints[:, :2] = (frame_keypoints[:, :2] - min_val) / (max_val - min_val)
        
        # Center in frame
        center_x = 0.5
        center_y = 0.5
        frame_keypoints[:, 0] = frame_keypoints[:, 0] - np.mean(frame_keypoints[:, 0]) + center_x
        frame_keypoints[:, 1] = frame_keypoints[:, 1] - np.mean(frame_keypoints[:, 1]) + center_y
        
        poses[i] = frame_keypoints.reshape(-1)
    
    return gloss_text, poses

st.title("Text to Sign Language Translator")
st.write("Enter text below to see its sign language translation")

t2g_model, g2p_model, dataset = load_models()

user_input = st.text_input("Input Text:", "Hello world")
generate_btn = st.button("Generate Sign Language")

if generate_btn and user_input:
    with st.spinner("Generating sign language..."):
        gloss_text, poses = text_to_sign(
            user_input, 
            t2g_model, 
            g2p_model, 
            dataset
        )
        
        # Display gloss sequence
        st.subheader("Gloss Sequence")
        st.write(gloss_text)
        
        # Create video
        st.subheader("Sign Language Animation")
        output_path = "output_video.mp4"

        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 5.0, (512, 512))

        # Render and write each frame
        for pose in poses:
            frame = render_pose_frame(pose)  # Must return a (512, 512, 3) RGB np.array
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        # Add extra frames to hold the last frame longer
        for _ in range(10):
            out.write(frame_bgr)

        # Finalize the video
        out.release()

        # Display the video in Streamlit
        if os.path.exists(output_path):
            st.video(output_path, format='video/mp4')
        else:
            st.error("Failed to generate video.")
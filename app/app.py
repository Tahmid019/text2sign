import streamlit as st
import torch
import numpy as np
import cv2
import tempfile
from models import Text2GlossTransformer, Gloss2Pose, TextGlossDataset3
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
POSE_CONNECTIONS = [
    (0,1), (0,2), (1,3), (2,4),        # Head
    (5,6), (5,7), (7,9), (6,8), (8,10), # Arms
    (11,12), (11,13), (13,15), (12,14), (14,16) # Legs
]

current_dir = Path(__file__).parent

@st.cache_resource
def load_models():
    """Load pre-trained models and dataset"""
    dataset = TextGlossDataset3(current_dir / 'processed_data.pt')
    
    t2g_model = Text2GlossTransformer(
        len(dataset.text_vocab),
        len(dataset.gloss_vocab)
    ).to(DEVICE)
    t2g_model.load_state_dict(torch.load(current_dir / 't2g_model_weights.pth', map_location=DEVICE))
    t2g_model.eval()
    
    g2p_model = Gloss2Pose(len(dataset.gloss_vocab)).to(DEVICE)
    g2p_model.load_state_dict(torch.load(current_dir / 'g2p_model_weights.pth', map_location=DEVICE))
    g2p_model.eval()
    
    return t2g_model, g2p_model, dataset

def render_pose_frame(pose, frame_size=(512, 512)):
    """Render a single pose frame"""
    frame = np.zeros((*frame_size, 3), dtype=np.uint8)
    keypoints = pose.reshape(-1, 3)
    keypoints[:, :2] = keypoints[:, :2] * frame_size[0]
    
    # Draw connections
    for i, j in POSE_CONNECTIONS:
        if keypoints[i, 2] > 0.2 and keypoints[j, 2] > 0.2:
            cv2.line(
                frame,
                (int(keypoints[i, 0]), int(keypoints[i, 1])),
                (int(keypoints[j, 0]), int(keypoints[j, 1])),
                (255, 166, 2), 2
            )
    
    # Draw points
    for x, y, conf in keypoints:
        if conf > 0.2:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
    
    return frame

def text_to_sign(text, t2g_model, g2p_model, dataset):
    """Convert text to sign language video"""
    # Tokenize input text
    tokens = [dataset.text_vocab.get(w.lower(), dataset.text_vocab["<unk>"]) 
              for w in text.split()]
    src_tokens = torch.tensor([tokens]).to(DEVICE)
    
    # Generate gloss sequence
    gloss_seq = [dataset.gloss_vocab["<sos>"]]
    for _ in range(20):  # Max output length
        decoder_input = torch.tensor([gloss_seq]).to(DEVICE)
        with torch.no_grad():
            logits = t2g_model(src_tokens, decoder_input)
        next_id = logits[0, -1].argmax().item()
        if next_id == dataset.gloss_vocab["<eos>"]:
            break
        gloss_seq.append(next_id)
    
    # Convert gloss IDs to text
    gloss_text = ' '.join([dataset.inv_gloss.get(idx, '<unk>') 
                          for idx in gloss_seq[1:]])
    
    # Generate poses
    gloss_tensor = torch.tensor([gloss_seq[1:]]).to(DEVICE)
    with torch.no_grad():
        poses = g2p_model(gloss_tensor).cpu().numpy()[0]
    
    # Normalize poses
    poses = (poses - np.min(poses)) / (np.max(poses) - np.min(poses) + 1e-8)
    
    return gloss_text, poses

# Streamlit UI
st.title("Text to Sign Language Translator")
st.write("Enter text below to see its sign language translation")

# Load models
t2g_model, g2p_model, dataset = load_models()

# User input
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
        video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_file.name, fourcc, 5.0, (512, 512))
        
        for pose in poses:
            frame = render_pose_frame(pose)
            out.write(frame)
        
        out.release()
        
        # Display video
        st.video(video_file.name)
        video_file.close()
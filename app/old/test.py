import torch
import numpy as np
import streamlit as st
import cv2
import tempfile
from collections import defaultdict

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Dataset Implementation
class DemoDataset:
    def __init__(self):
        # Vocabulary setup
        self.text_vocab = defaultdict(lambda: len(self.text_vocab))
        self.gloss_vocab = defaultdict(lambda: len(self.gloss_vocab))
        
        # Special tokens
        self.text_vocab["<unk>"] = 0
        self.text_vocab["<sos>"] = 1
        self.text_vocab["<eos>"] = 2
        self.gloss_vocab["<sos>"] = 0
        self.gloss_vocab["<eos>"] = 1
        
        # Add words we know
        for word in ["hello", "world"]:
            self.text_vocab[word.lower()]
        
        # Add glosses we know
        for gloss in ["HELLO", "WORLD"]:
            self.gloss_vocab[gloss]
        
        # Create inverse mappings
        self.inv_gloss = {v: k for k, v in self.gloss_vocab.items()}
        
        # Simple mapping from text to glosses
        self.text_to_gloss = {
            "hello": ["HELLO"],
            "world": ["WORLD"]
        }
    
    def __len__(self):
        return 1  # Just for demo

# 2. Text-to-Gloss Model
class TextToGlossModel(torch.nn.Module):
    def __init__(self, text_vocab_size, gloss_vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(text_vocab_size, 16)
        self.encoder = torch.nn.LSTM(16, 32, batch_first=True)
        self.decoder = torch.nn.LSTM(16, 32, batch_first=True)
        self.fc = torch.nn.Linear(32, gloss_vocab_size)
        
    def forward(self, src_tokens, decoder_input):
        # Encoder
        src_embed = self.embedding(src_tokens)
        enc_out, (hidden, cell) = self.encoder(src_embed)
        
        # Decoder
        dec_embed = self.embedding(decoder_input)
        dec_out, _ = self.decoder(dec_embed, (hidden, cell))
        
        # Project to gloss vocabulary
        logits = self.fc(dec_out)
        return logits

def load_t2g_weights(model, dataset):
    with torch.no_grad():
        # Initialize embeddings
        torch.nn.init.normal_(model.embedding.weight, 0, 0.1)
        
        # Initialize LSTM weights
        for name, param in model.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        
        # Create direct mappings
        hello_idx = dataset.text_vocab["hello"]
        world_idx = dataset.text_vocab["world"]
        
        # Map "hello" to HELLO gloss
        model.fc.weight.data[dataset.gloss_vocab["HELLO"]] = torch.zeros(32)
        model.fc.bias.data[dataset.gloss_vocab["HELLO"]] = 10
        
        # Map "world" to WORLD gloss
        model.fc.weight.data[dataset.gloss_vocab["WORLD"]] = torch.zeros(32)
        model.fc.bias.data[dataset.gloss_vocab["WORLD"]] = 10
        
    return model

# 3. Gloss-to-Pose Model (simplified version)
class GlossToPoseModel(torch.nn.Module):
    def __init__(self, gloss_vocab_size, pose_dim=51):
        super().__init__()
        # Simple lookup table that maps gloss IDs to pose vectors
        self.embedding = torch.nn.Embedding(gloss_vocab_size, pose_dim)
        
    def forward(self, gloss_tensor):
        # Directly return pose vectors for gloss IDs
        return self.embedding(gloss_tensor)

def load_g2p_weights(model, dataset):
    with torch.no_grad():
        # Create simple poses for our glosses
        def make_pose(hand_x, hand_y):
            # Simple pose with hand at specified position
            pose = np.zeros(51)
            
            # Shoulder at (0.5, 0.3)
            pose[0:3] = [0.5, 0.3, 1]  # Left shoulder
            pose[3:6] = [0.5, 0.3, 1]  # Right shoulder
            
            # Elbow at (0.5, 0.5)
            pose[6:9] = [0.5, 0.5, 1]  # Left elbow
            pose[9:12] = [0.5, 0.5, 1]  # Right elbow
            
            # Hand at specified position
            pose[12:15] = [hand_x, hand_y, 1]  # Left hand
            pose[15:18] = [hand_x, hand_y, 1]  # Right hand
            
            # Other keypoints (face, etc.) at neutral positions
            for i in range(18, 51, 3):
                pose[i:i+2] = [0.5, 0.5]
                pose[i+2] = 1
                
            return pose.astype(np.float32)
        
        # Set poses for specific glosses
        model.embedding.weight.data[dataset.gloss_vocab["HELLO"]] = torch.tensor(make_pose(0.7, 0.7))
        model.embedding.weight.data[dataset.gloss_vocab["WORLD"]] = torch.tensor(make_pose(0.3, 0.7))
        
        # Set neutral pose for other glosses
        neutral_pose = torch.tensor(make_pose(0.5, 0.5))
        for i in range(len(dataset.gloss_vocab)):
            if i not in [dataset.gloss_vocab["HELLO"], dataset.gloss_vocab["WORLD"]]:
                model.embedding.weight.data[i] = neutral_pose
        
    return model

# 4. Pose Rendering Function
def render_pose_frame(pose):
    # Create blank image
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Reshape pose to keypoints (17 points with x,y,visibility)
    keypoints = pose.reshape(-1, 3)
    
    # Define connections between keypoints
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),     # Face
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Body
        (11, 13), (13, 15),                  # Left arm
        (12, 14), (14, 16)                   # Right arm
    ]
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start = keypoints[start_idx]
            end = keypoints[end_idx]
            if start[2] > 0.1 and end[2] > 0.1:  # Only draw if visible
                cv2.line(img, 
                        (int(start[0]*512), int(start[1]*512)),
                        (int(end[0]*512), int(end[1]*512)),
                        (0, 0, 255), 2)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if kp[2] > 0.1:  # If visible
            color = (0, 255, 0) if i in [15, 16] else (0, 0, 255)  # Hands in green
            cv2.circle(img, (int(kp[0]*512), int(kp[1]*512)), 6, color, -1)
    
    return img

def text_to_sign(text, t2g_model, g2p_model, dataset):
    """Convert text to sign language video with proper pose normalization"""
    # Tokenize input text
    tokens = [dataset.text_vocab.get(w.lower(), dataset.text_vocab["<unk>"]) 
              for w in text.split()]
    tokens = [dataset.text_vocab["<sos>"]] + tokens + [dataset.text_vocab["<eos>"]]
    src_tokens = torch.tensor([tokens]).to(DEVICE)
    
    # Generate gloss sequence
    gloss_seq = [dataset.gloss_vocab["<sos>"]]
    for i in range(50):  # Max output length
        decoder_input = torch.tensor([gloss_seq]).to(DEVICE)
        with torch.no_grad():
            logits = t2g_model(src_tokens, decoder_input)
        next_id = logits[0, -1].argmax().item()
        if next_id == dataset.gloss_vocab["<eos>"]:
            break
        gloss_seq.append(next_id)
    
    # Convert gloss IDs to text
    gloss_text = ' '.join([dataset.inv_gloss.get(idx, '<unk>') 
                          for idx in gloss_seq[1:-1]])  # Remove <sos> and <eos>
    
    # Generate poses
    gloss_ids = torch.tensor(gloss_seq[1:-1]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        poses = g2p_model(gloss_ids).cpu().numpy()[0]
    
    # Normalize poses
    for i in range(len(poses)):
        frame_keypoints = poses[i].reshape(-1, 3)
        
        # Find min/max for normalization
        min_x, min_y = frame_keypoints[:, 0].min(), frame_keypoints[:, 1].min()
        max_x, max_y = frame_keypoints[:, 0].max(), frame_keypoints[:, 1].max()
        
        # Normalize to [0,1] range
        if max_x - min_x > 1e-8:
            frame_keypoints[:, 0] = (frame_keypoints[:, 0] - min_x) / (max_x - min_x)
        if max_y - min_y > 1e-8:
            frame_keypoints[:, 1] = (frame_keypoints[:, 1] - min_y) / (max_y - min_y)
        
        # Center in frame
        frame_keypoints[:, 0] = frame_keypoints[:, 0] - np.mean(frame_keypoints[:, 0]) + 0.5
        frame_keypoints[:, 1] = frame_keypoints[:, 1] - np.mean(frame_keypoints[:, 1]) + 0.5
        
        poses[i] = frame_keypoints.reshape(-1)
    
    return gloss_text, poses

# 6. Model Loading Function
def load_models():
    # Initialize dataset and models
    dataset = DemoDataset()
    t2g_model = TextToGlossModel(len(dataset.text_vocab), len(dataset.gloss_vocab)).to(DEVICE)
    g2p_model = GlossToPoseModel(len(dataset.gloss_vocab)).to(DEVICE)
    
    # Load simple weights
    t2g_model = load_t2g_weights(t2g_model, dataset)
    g2p_model = load_g2p_weights(g2p_model, dataset)
    
    return t2g_model, g2p_model, dataset

# 7. Streamlit App
def main():
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
            
            if len(poses) == 0:
                st.error("No poses generated. Please try different text.")
                return
                
            # Create video
            st.subheader("Sign Language Animation")
            video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_file.name, fourcc, 5.0, (512, 512))
            
            # Render each frame
            for pose in poses:
                frame = render_pose_frame(pose)
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            # Add extra frames for longer display
            for _ in range(10):
                out.write(frame_bgr)
            
            out.release()
            
            # Display video with controls
            video_bytes = open(video_file.name, 'rb').read()
            st.video(video_bytes, format='video/mp4')
            video_file.close()

if __name__ == "__main__":
    main()
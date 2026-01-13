import os
import json
import torch
import tempfile
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import StackedBiLSTMTransformerModel, PreProcessor

# --- NEW IMPORTS ---
import cv2
import mediapipe as mp
import numpy as np

# --- Configuration and Initialization ---

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {DEVICE}")

try:
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("error: config.json not found. please create it.")
    exit()

MODEL = StackedBiLSTMTransformerModel
PRE_PROCESSOR = PreProcessor
custom_temp_dir = r"C:\Users\tahmi\Documents\Work\Text2Sign\t2slt\backends\s2t\assets\videos"

def load_model_and_config():
    """Load model + preprocessor"""
    global MODEL, PRE_PROCESSOR
    print("\nloading model and assets...")

    try:
        PRE_PROCESSOR = PreProcessor(
            stats_path=CONFIG['stats_path'],
            label_map_path=CONFIG['label_map_path']
        )
        print("  pre-processor initialized.")
    except Exception as e:
        print(f"error initializing pre-processor: {e}")
        return

    try:
        model_args = CONFIG['model_params']
        MODEL = StackedBiLSTMTransformerModel(**model_args).to(DEVICE)
    except Exception as e:
        print(f"error initializing model architecture: {e}")
        return

    try:
        checkpoint = torch.load(CONFIG['model_path'], map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.eval()
        print(f"  model loaded successfully from {CONFIG['model_path']}")
    except FileNotFoundError:
        print(f"error: model checkpoint not found at {CONFIG['model_path']}")
    except Exception as e:
        print(f"error loading model weights: {e}")


load_model_and_config()

app = Flask(__name__)
CORS(app)

# --- Landmark Extraction Function ---

def flatten_landmarks(landmarks):
    """Flatten a list of MediaPipe landmarks into [x, y, z, visibility, ...]."""
    if landmarks is None:
        return []
    return [
        coord
        for lm in landmarks.landmark
        for coord in (lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.0))
    ]

def extract_landmarks_from_video(video_path):
    """
    Extract holistic landmarks (pose + face + hands) from a video file.
    Returns a list of frames, each a dict containing flattened landmarks.
    """
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    all_frames = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (640, 480))
        results = holistic.process(frame_rgb)

        # collect all landmark data
        frame_data = {
            'face': flatten_landmarks(results.face_landmarks),
            'pose': flatten_landmarks(results.pose_landmarks),
            'left_hand': flatten_landmarks(results.left_hand_landmarks),
            'right_hand': flatten_landmarks(results.right_hand_landmarks)
        }
        # print("Left hand landmarks:", results.left_hand_landmarks)
        # print("Right hand landmarks:", results.right_hand_landmarks)

        all_frames.append(frame_data)

    cap.release()
    holistic.close()
    return all_frames


# --- Predict Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if MODEL is None or PRE_PROCESSOR is None:
            print("Error: Model or PreProcessor not loaded")
            return jsonify({"status": "error", "message": "Model not loaded"}), 500

        if 'video' not in request.files:
            print("Error: No video file uploaded")
            return jsonify({"status": "error", "message": "No video file uploaded"}), 400

        video_file = request.files['video']
        print("Received video:", video_file.filename)

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=custom_temp_dir) as tmp:
            video_path = tmp.name
            video_file.save(video_path)
        print("Saved video to:", video_path)

        cap = cv2.VideoCapture(video_path)
        print("cap.isOpened():", cap.isOpened())
        if not cap.isOpened():
            raise ValueError("OpenCV cannot open the video file!")

        raw_frames = extract_landmarks_from_video(video_path)
        print("Sample frame:", raw_frames[0])
        print("Sum of features:", np.sum(raw_frames[0]['pose']))
        print("Frames extracted:", len(raw_frames))
        if len(raw_frames) < 5:
            raise ValueError("Too few frames detected. Need at least 5.")



        try:
            data_tensor = PRE_PROCESSOR.preprocess(raw_frames).to(DEVICE)
            print("Data tensor shape:", data_tensor.shape)
        except Exception as e:
            print("PreProcessor error:", e)
            raise e

        with torch.no_grad():
            outputs = MODEL(data_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
            top_prob, top_idx = torch.max(probs, 0)
            gloss = PRE_PROCESSOR.idx_to_gloss.get(top_idx.item(), f"unknown_{top_idx.item()}")
            print(f"Prediction: {gloss}, Confidence: {top_prob.item():.4f}")

        return jsonify({
            "status": "success",
            "prediction_gloss": gloss,
            "confidence": f"{top_prob.item():.4f}"
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        if 'cap' in locals():
            cap.release()
            print("VideoCapture released")

        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
            print("Temporary video file removed:", video_path)





# --- Run Server ---

# --- Run Server ---
if __name__ == '__main__':
    print("\n--- API Ready ---")
    print("POST a video to: http://127.0.0.1:5000/predict")
    app.run(host='0.0.0.0', port=5000, debug=True)


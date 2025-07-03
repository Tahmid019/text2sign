import os
import json
import cv2
import imageio
import numpy as np
from pathlib import Path
from mediapipe import solutions
from tqdm import tqdm

# ---------- CONFIGURATION ----------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / 'datasets' / 'ISL_Gifs'
GLOSS_MAP_PATH = REPO_ROOT / 'datasets' / 'invGlossList.json'
OUTPUT_PATH = REPO_ROOT / 'datasets' / 'augmented_gloss_landmarks.json'
MAX_SEQ_LEN = 100
IMAGE_SIZE = (256, 256)
VALID_VIDEO_EXTS = ['.gif', '.mp4', '.avi']
VALID_IMAGE_EXTS = ['.jpg', '.jpeg', '.png']

# ---------- LANDMARK PROCESSOR ----------
holistic = solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- HELPER FUNCTIONS ----------
def load_frames(path: Path):
    """
    Load frames from a video file (gif/mp4/avi) or a directory of images.
    Returns a list of BGR images resized to IMAGE_SIZE.
    """
    frames = []
    if path.suffix.lower() in VALID_VIDEO_EXTS:
        reader = imageio.get_reader(str(path))
        for frame in reader:
            frames.append(cv2.resize(frame, IMAGE_SIZE))
        reader.close()
    elif path.is_dir():
        for img_file in sorted(path.iterdir()):
            if img_file.suffix.lower() in VALID_IMAGE_EXTS:
                img = cv2.imread(str(img_file))
                if img is not None:
                    frames.append(cv2.resize(img, IMAGE_SIZE))
    else:
        raise FileNotFoundError(f"Unsupported path or missing file: {path}")
    return frames

def extract_landmarks(frames: list) -> list:
    """
    Extract left-hand, right-hand, and pose landmarks from each frame.
    Returns a list of length MAX_SEQ_LEN, each element a list of floats.
    """
    seq = []
    for img in frames:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)
        vec = []
        # left hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 21 * 3)
        # right hand
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 21 * 3)
        # body pose
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 33 * 3)
        seq.append(vec)
    # pad / trim sequence
    arr = np.array(seq, dtype=np.float32)
    if arr.shape[0] < MAX_SEQ_LEN:
        pad = np.zeros((MAX_SEQ_LEN - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.vstack((arr, pad))
    else:
        arr = arr[:MAX_SEQ_LEN]
    return arr.tolist()

# ---------- MAIN ----------
def main():
    # Load gloss map
    with open(GLOSS_MAP_PATH, 'r') as f:
        gloss_map = json.load(f)

    augmented = {}
    for gloss, vids in gloss_map.items():
        augmented[gloss] = []
        for vid in tqdm(vids, desc=f"Processing gloss '{gloss}'"):
            sample_path = None
            for ext in VALID_VIDEO_EXTS:
                candidate = DATA_DIR / f"{vid}{ext}"
                if candidate.exists():
                    sample_path = candidate
                    break
            if sample_path is None:
                dir_candidate = DATA_DIR / vid
                if dir_candidate.is_dir():
                    sample_path = dir_candidate
            if sample_path is None:
                print(f"[WARNING] Sample not found for ID={vid}")
                continue

            # load frames and extract landmarks
            try:
                frames = load_frames(sample_path)
                landmarks = extract_landmarks(frames)
                augmented[gloss].append({
                    'vid_id': vid,
                    'landmarks': landmarks
                })
            except Exception as e:
                print(f"Error processing {sample_path}: {e}")

    with open(OUTPUT_PATH, 'w') as out:
        json.dump(augmented, out, indent=2)
    print(f"Augmented dataset saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()

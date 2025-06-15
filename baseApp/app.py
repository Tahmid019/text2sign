import streamlit as st
import os
import json
# from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp

def generate_keyframe_skeleton(video_path, out_dir="archive/keyframe", threshold=1.0):
    import cv2
    import os
    import mediapipe as mp
    
    os.makedirs(out_dir, exist_ok=True)
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{video_id}.mp4")

    if os.path.exists(out_path):
        print(f"[SKIP] Keyframe video already exists: {out_path}")
        return out_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Initialize MediaPipe components
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        writer.release()
        raise IOError("Couldn't read first frame.")
    
    # Process and write first frame
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(first_frame_rgb)
    blank = np.zeros_like(first_frame)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(blank, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(blank, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(blank, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    writer.write(blank)

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3,
            winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = mag.mean()

        if mean_mag > threshold:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            blank = np.zeros_like(frame)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    blank, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    blank, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    blank, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            writer.write(blank)
            frame_count += 1

        prev_gray = gray

    if frame_count == 0:
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
        print(f"[WARNING] No motion detected. Created placeholder frame.")

    holistic.close()
    cap.release()
    writer.release()

    print(f"[DONE] Saved {frame_count} keyframes to {out_path}")
    return out_path



with open('archive/WLASL_v0.3.json', 'r') as f:
    wlasl_data = json.load(f)

gloss_to_videos = {}
for entry in wlasl_data:
    gloss = entry['gloss'].lower()
    gloss_to_videos[gloss] = []
    for inst in entry['instances']:
        gloss_to_videos[gloss].append({
            'video_id': inst['video_id'],
            'url': inst['url']
        })

missing_path = 'archive/missing.txt'
if os.path.exists(missing_path):
    with open(missing_path, 'r') as f:
        missing_ids = set(line.strip() for line in f)
else:
    missing_ids = set()

def get_video_paths(text):
    word = text.strip().lower()
    videos = gloss_to_videos.get(word, [])
    available_videos = []
    for v in videos:
        video_file = os.path.join('archive/videos', f"{v['video_id']}.mp4")
        if v['video_id'] not in missing_ids and os.path.exists(video_file):
            available_videos.append(video_file)
    return available_videos

st.title("Text to Sign Language Video")

user_text = st.text_input("Enter text:")

if user_text:
    main_video_path = get_video_paths(user_text)
    video_path = generate_keyframe_skeleton(get_video_paths(user_text)[0])
    if video_path:
        st.video(main_video_path[0])
        st.video(video_path)
    else:
        st.error(f"No video found for: {user_text}")
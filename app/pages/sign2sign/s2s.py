import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from keras.models import load_model



from config import PROJECT_ROOT, APP_ROOT, MODEL_DIR
MODEL_PATH = f"{PROJECT_ROOT}/{APP_ROOT}/{MODEL_DIR}/Final.h5"
ACTIONS = np.array(__import__('os').listdir(f"{PROJECT_ROOT}/{APP_ROOT}/{MODEL_DIR}"))

from .KeypointsExtraction import keypoint_extraction, image_process, draw_landmarks

def init_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    return engine

class WebRTCSignDetector(VideoTransformerBase):
    def __init__(self):
        # Load model and TTS
        self.model = load_model(MODEL_PATH)
        self.engine = init_tts_engine()
        self.last_prediction = None
        # MediaPipe holistic setup
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        # Buffers and state
        self.frame_buffer = []
        self.sentence = []
        self.cooldown = 0
        self.cooldown_thresh = 20
        self.skip_after_detect = 5
        self.skip_counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Process landmarks
        results = image_process(img, self.holistic)
        draw_landmarks(img, results)

        hand_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)
        # Skip initial frames on detection
        if hand_detected:
            if self.skip_counter < self.skip_after_detect:
                self.skip_counter += 1
            else:
                self.frame_buffer.append(keypoint_extraction(results))
        else:
            self.skip_counter = 0
            self.frame_buffer.clear()

        # Predict every 20 frames
        if len(self.frame_buffer) >= 20 and self.cooldown == 0:
            data = np.array(self.frame_buffer)[np.newaxis, :, :]
            preds = self.model.predict(data)
            self.frame_buffer.clear()
            if np.max(preds) >= 0.85:
                idx = np.argmax(preds)
                action = ACTIONS[idx]
                if action != self.last_prediction:
                    self.sentence.append(action)
                    self.last_prediction = action
                    self.engine.say(action)
                    self.engine.runAndWait()
                    self.cooldown = self.cooldown_thresh
        # Cooldown decrement
        self.cooldown = max(0, self.cooldown - 1)

        # Display the last 7 words
        if self.sentence:
            text = ' '.join(self.sentence[-7:]).capitalize()
            cv2.putText(img, text, (10, img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    
def s2s():
    st.title("SIgn-2-Sign Interface")
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("SL2T: Real-Time Sign to Text ")
        
        text_placeholder = st.empty()
        
        webrtc_ctx = webrtc_streamer(
            key="sl2t_stream",
            video_transformer_factory=WebRTCSignDetector,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
            text_placeholder.markdown(
                f"### Detected: `{getattr(webrtc_ctx.video_transformer, 'last_prediction', '')}`"
            )


    # --- Right Column: T2SL Text to Sign ---
    with right_col:
        st.subheader("T2SL: Text to Sign ")
        user_input = st.text_input("Enter text to translate into Sign Language:")
        if st.button("Generate Sign Video") and user_input:
            try:
                from pages.text2sign.t2sl import t2sl
                video_path = t2sl(user_input)
                st.video(video_path)
                st.markdown(f"**Input Text:** {user_input}")
            except Exception as e:
                st.error(f"Error: {e}")

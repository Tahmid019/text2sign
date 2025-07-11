import os
import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
import mediapipe as mp
from keras.models import Sequential
from keras.layers import LSTM, Dense

from config import *

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils



# Load model (uncomment if you have a saved model)
# model = load_model('action.h5')

# Or define the model architecture (if you don't have saved .h5)
@st.cache_resource
def create_model():
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model

model = create_model()
model.load_weights('app/models/LSTM_Model1.h5')  # Load weights if available

# 1. New detection variables


# 2. Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# 3. Function to draw styled landmarks
def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

# 4. Function for mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION
    image.flags.writeable = False                  # Image no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION
    return image, results



# Set mediapipe model 
def s2t_main(incomming_text = ""):
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8
    
    # Initialize webcam
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    TEXT_WINDOW = st.empty()

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while run:
            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                # 3. Visualization logic
                if np.unique(predictions[-10:])[0] == np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]
                
                incomming_text = sentence
                GLOBAL_CURR_TEXT = sentence
                

                # Viz probabilities
                image = cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                image = cv2.putText(image, ' '.join(sentence), (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display prediction probabilities
                for i, prob in enumerate(res):
                    text = f"{actions[i]}: {prob:.2f}"
                    cv2.putText(image, text, (10, 60 + i*30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            # Show to screen
            FRAME_WINDOW.image(image)
            TEXT_WINDOW.text(f"Current Prediction: {' '.join(sentence) if sentence else 'None'}")

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        return incomming_text
        
def s2t(in_text = ""):
    # Streamlit UI
    st.title("Real-Time Sign Language Detection")
    st.markdown("Detecting actions: Hello, Thanks, I Love You")
    
    s2t_main(in_text)
    
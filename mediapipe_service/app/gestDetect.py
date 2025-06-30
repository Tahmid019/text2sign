from imports import *
from config import *
from KeypointsExtraction import *

def sl2t():
    # Initialize Text-to-Speech Engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speech rate (optional)
    engine.setProperty('volume', 1.0)  # Max volume

    # Path to data and actions defined during training
    PATH = os.path.join(f'{PROJECT_ROOT}/{APP_ROOT}/{MODEL_DIR}')
    actions = np.array(os.listdir(PATH))

    # Load the trained model
    model = load_model(f'{PROJECT_ROOT}/{APP_ROOT}/{MODEL_DIR}/Final.h5')

    # Initialize prediction and sentence-related lists
    sentence, keypoints, last_prediction = [], [], None
    cooldown_frames, cooldown_threshold = 0, 20  # Cooldown period of 20 frames after each prediction
    skip_frames_after_hand_detected, skip_counter = 5, 0  # Skip 5 frames after hand is detected

    # Open camera for capturing
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        exit()

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.70, min_tracking_confidence=0.70) as holistic:
        hand_present = False  # Track if a hand is present

        while cap.isOpened():
            # Capture frame from camera
            ret, image = cap.read()
            if not ret:
                break

            # Process frame and extract keypoints
            results = image_process(image, holistic)
            draw_landmarks(image, results)

            # Check if a hand is present in the frame
            hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

            if hand_detected:
                if not hand_present:
                    # Hand just appeared, start skip counter
                    hand_present = True
                    skip_counter = skip_frames_after_hand_detected
                elif skip_counter > 0:
                    skip_counter -= 1
                    continue

                # Extract keypoints after hand has been stable for 5 frames
                keypoints.append(keypoint_extraction(results))

                # Predict every 20 frames if cooldown is not active
                if len(keypoints) == 20 and cooldown_frames == 0:
                    keypoints = np.array(keypoints)
                    prediction = model.predict(keypoints[np.newaxis, :, :])
                    keypoints = []

                    if np.max(prediction) >= 0.85:
                        predicted_action = actions[np.argmax(prediction)]

                        if predicted_action != last_prediction:
                            sentence.append(predicted_action)
                            last_prediction = predicted_action
                            cooldown_frames = cooldown_threshold

                            # 🔊 Speak the predicted phrase
                            engine.say(predicted_action)
                            engine.runAndWait()

            else:
                hand_present = False
                keypoints = []

            cooldown_frames = max(0, cooldown_frames - 1)

            if len(sentence) > 7:
                sentence = sentence[-7:]

            if keyboard.is_pressed(' '):
                sentence, keypoints, last_prediction = [], [], None

            if sentence:
                sentence[0] = sentence[0].capitalize()

            # Display sentence
            display_text = ' '.join(sentence)
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            cv2.putText(image, display_text, (text_x, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Real-time Sign Prediction', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release everything
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    sl2t()
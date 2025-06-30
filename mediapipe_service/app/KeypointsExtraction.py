from imports import *


def image_process(image, holistic):
    """
    Processes an image frame using MediaPipe Holistic
    and returns detection results.
    
    Args:
        image: Input frame from camera
        holistic: Initialized MediaPipe Holistic model
        
    Returns:
        MediaPipe Holistic results object
    """
    # Convert BGR to RGB and process with MediaPipe
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):
    """
    Extracts and concatenates keypoints from Holistic results
    into a flattened numpy array.
    
    Args:
        results: MediaPipe Holistic results object
        
    Returns:
        Flattened numpy array of keypoint coordinates (x,y,z)
    """
    # Initialize empty arrays for each component
    face = np.array([[res.x, res.y, res.z] for res in 
                    results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    pose = np.array([[res.x, res.y, res.z] for res in 
                    results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    
    lh = np.array([[res.x, res.y, res.z] for res in 
                  results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    rh = np.array([[res.x, res.y, res.z] for res in 
                  results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

def draw_landmarks(image, results):
    """Visualizes landmarks on the image"""
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)


import json
import numpy as np
import cv2
import mediapipe as mp

# =======
INPUT_JSON = r"E:\Balanced_20_Frames_Augmented\Train\carrot\09309\landmarks.json"     
OUTPUT_VIDEO = "carrot_1_new.mp4"

IMG_SIZE = 800
FPS = 15

DRAW_FACE = False      
MAX_DIST = 120          

# =========
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

POSE_CONN = mp_pose.POSE_CONNECTIONS
HAND_CONN = mp_hands.HAND_CONNECTIONS
FACE_CONN = mp_face.FACEMESH_CONTOURS

def clip_and_scale(xy):
    xy = np.nan_to_num(xy)
    xy = np.clip(xy, 0.0, 1.0)
    return (xy * IMG_SIZE).astype(np.int32)

def safe_line(img, p1, p2, color, thickness):
    if np.linalg.norm(p1 - p2) < MAX_DIST:
        cv2.line(img, tuple(p1), tuple(p2), color, thickness)

# ====
with open(INPUT_JSON, "r") as f:
    frames = json.load(f)

print(f"[INFO] Loaded {len(frames)} frames")

def safe_points(data, expected_len):
    """
    Safely convert landmark list to (N,2) numpy array.
    Returns None if data is invalid.
    """
    if data is None:
        return None
    if not isinstance(data, (list, tuple)):
        return None
    if len(data) != expected_len:
        return None

    arr = np.array(data, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] < 2:
        return None

    return arr[:, :2]


# =============
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    FPS,
    (IMG_SIZE, IMG_SIZE)
)

# ==============
for t, frame_data in enumerate(frames):
    canvas = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

    # ---------- Pose ----------
    pose = safe_points(frame_data.get("pose"), 33)
    if pose is not None:
        pose = clip_and_scale(pose)
        for a, b in POSE_CONN:
            safe_line(canvas, pose[a], pose[b], (0,0,255), 2)
        for p in pose:
            cv2.circle(canvas, tuple(p), 3, (0,0,200), -1)

    # ---------- Left hand ----------
    left = safe_points(frame_data.get("left_hand"), 21)
    if left is not None:
        left = clip_and_scale(left)
        for a, b in HAND_CONN:
            safe_line(canvas, left[a], left[b], (255,0,0), 2)
        for p in left:
            cv2.circle(canvas, tuple(p), 3, (200,0,0), -1)

    # ---------- Right hand ----------
    right = safe_points(frame_data.get("right_hand"), 21)
    if right is not None:
        right = clip_and_scale(right)
        for a, b in HAND_CONN:
            safe_line(canvas, right[a], right[b], (0,255,0), 2)
        for p in right:
            cv2.circle(canvas, tuple(p), 3, (0,200,0), -1)

    # ---------- Face (optional) ----------
    if DRAW_FACE:
        face = safe_points(frame_data.get("face"), 468)
        if face is not None:
            face = clip_and_scale(face)
            for a, b in FACE_CONN:
                safe_line(canvas, face[a], face[b], (180,180,180), 1)

    writer.write(canvas)

writer.release()
print(f"[DONE] Video saved to {OUTPUT_VIDEO}")

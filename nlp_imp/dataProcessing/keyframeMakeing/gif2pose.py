import cv2
import time
from ...comp_vision.poseEstimation import poseModule as pm
import os

def generatePoseVid(vid_pth):
    ext = os.path.splitext(vid_pth)[1].lower()
    base_name = os.path.splitext(os.path.basename(vid_pth))[0]

    if ext == ".jpg":
        img = cv2.imread(vid_pth)
        if img is None:
            print(f"[!] Could not read image: {vid_pth}")
            return

        detector = pm.poseDetector()
        img_out = detector.findPose(img)
        #detector.findPosition(img_out, draw=True) 
        out_path = f'images/output_{base_name}.jpg'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img_out)
        print(f"[+] Wrote processed image to {out_path}")
        return

    if ext not in (".mp4", ".gif"):
        print(f"Skipping unsupported file type: {vid_pth}")
        return

    cap = cv2.VideoCapture(vid_pth)
    if not cap.isOpened():
        print(f"[!] Failed to open video: {vid_pth}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 10  # fallback fps if 0

    os.makedirs('videos', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = f'videos/output_{base_name}.mp4'
    out = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))

    pTime = 0
    detector = pm.poseDetector()
    while True:
        success, img = cap.read()
        if not success or img is None:
            break

        img = detector.findPose_onlyframe(img)
        # detector.findPosition(img, draw=True))
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime else 0
        pTime = cTime
        # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        out.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[+] Wrote processed video to {out_path}")

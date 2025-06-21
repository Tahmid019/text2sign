import cv2
import time
import nlp_imp.comp_vision.poseEstimation.poseModule as pm
import os

def generatePoseVid(vid_pth):
    cap = cv2.VideoCapture(vid_pth)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    base_name = os.path.splitext(os.path.basename(vid_pth))[0]
    out_name = f'videos/output_{base_name}.mp4'
    out = cv2.VideoWriter(out_name, fourcc, fps_in, (w, h))

    pTime = 0
    detector = pm.poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        # if len(lmList) !=0:
        #     print(lmList[14])
        #     cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        
        out.write(img)
        # cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


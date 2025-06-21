
from gif2pose import generatePoseVid
from ..datasets import ISL_Gifs
vid_dir = ISL_Gifs

for vid in vid_dir:
    generatePoseVid(vid)
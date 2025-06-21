from ..datasets import ISL_Gifs
import os
from tqdm import tqdm
from .gif2pose import generatePoseVid

if hasattr(ISL_Gifs, "__path__"):
    base_dir = ISL_Gifs.__path__[0]
else:
    base_dir = ISL_Gifs

allowed_exts = (".mp4", ".gif", ".jpg")
vid_list = [
    fname for fname in os.listdir(base_dir)
    if fname.lower().endswith(allowed_exts)
]

idx = 0
cnt = 10

for vid in tqdm(vid_list, desc="Processing videos"):
    idx = idx + 1
    # if cnt < idx: break
    vid_path = os.path.join(base_dir, vid)
    print(f"→ Processing {vid_path!r}")
    generatePoseVid(vid_path)

import os
import json
import shutil
from collections import defaultdict

# ===== CONFIG =====
ROOT = r"E:\WLASL\wlasl_full"
JSON_PATH = os.path.join(ROOT, "WLASL_v0.3.json")
VIDEOS_DIR = os.path.join(ROOT, "videos")

SAVE_ROOT = r"E:\WLASL\wlasl_1000_preprocessed"

K = 1000

# ==================

VIDEOS_SAVE_DIR = os.path.join(SAVE_ROOT, "videos")
os.makedirs(VIDEOS_SAVE_DIR, exist_ok=True)

# Load dataset (LIST)
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# ===== STEP 1: SELECT TOP-K GLOSSES (ALPHABETICAL) =====

all_glosses = sorted(set(entry["gloss"] for entry in data))
selected_glosses = all_glosses[:K]

gloss_to_id = {g: i+1 for i, g in enumerate(selected_glosses)}
LABEL_MAP = {i+1: g for i, g in enumerate(selected_glosses)}

print(f"Selected {len(LABEL_MAP)} gloss classes")

# ===== STEP 2: CREATE CLASS FOLDERS =====

for class_id in LABEL_MAP.keys():
    os.makedirs(os.path.join(VIDEOS_SAVE_DIR, str(class_id)), exist_ok=True)

# ===== STEP 3: PROCESS VIDEOS =====

# split → class_id → list of videos
subset_grouped = defaultdict(lambda: defaultdict(list))

for entry in data:
    gloss = entry["gloss"]

    if gloss not in gloss_to_id:
        continue

    class_id = gloss_to_id[gloss]

    for inst in entry["instances"]:
        vid = inst["video_id"]
        split = inst["split"]

        src = os.path.join(VIDEOS_DIR, f"{vid}.mp4")
        dst = os.path.join(VIDEOS_SAVE_DIR, str(class_id), f"{vid}.mp4")

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Missing video: {vid}")

        subset_grouped[split][class_id].append(vid)

# ===== STEP 4: SAVE LABEL FILES =====

# label.txt
with open(os.path.join(SAVE_ROOT, "label.txt"), "w") as f:
    for k in sorted(LABEL_MAP.keys()):
        f.write(f"{k}\t{LABEL_MAP[k]}\n")

# label_map_final.json
with open(os.path.join(SAVE_ROOT, "label_map_final.json"), "w") as f:
    json.dump(LABEL_MAP, f, indent=2)

# ===== STEP 5: SAVE SPLIT JSONS =====

for split, class_data in subset_grouped.items():
    output = []

    for class_id, vids in class_data.items():
        output.append({
            "id": str(class_id),
            "gloss": LABEL_MAP[class_id],
            "instances": [{"video_id": v} for v in vids]
        })

    out_path = os.path.join(SAVE_ROOT, f"{split}_final.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

print("✅ Done!")
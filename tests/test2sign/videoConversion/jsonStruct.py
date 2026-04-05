import os
import json

# -----------------------------
# Paths
# -----------------------------
LABEL_MAP_JSON = r"E:\Balanced_20_Frames_Augmented\NPY\label_map_final.json"
DATA_ROOT = r"E:\Balanced_20_Frames_Augmented\NPY"
OUTPUT_JSON = r"E:\Balanced_20_Frames_Augmented\NPY\train_final.json"

# -----------------------------
# Load label map (id -> gloss)
# -----------------------------
with open(LABEL_MAP_JSON, "r", encoding="utf-8") as f:
    label_map = json.load(f)   # e.g. { "1": "about", "2": "accident", ... }

annotations = []
total_samples = 0

# -----------------------------
# Scan folders by ID
# -----------------------------
for class_id, gloss in label_map.items():
    class_dir = os.path.join(DATA_ROOT, class_id)

    if not os.path.isdir(class_dir):
        print(f"[WARN] Folder not found: {class_dir}")
        continue

    instances = []
    for fname in sorted(os.listdir(class_dir)):
        if not fname.endswith(".npy"):
            continue

        video_id = os.path.splitext(fname)[0]
        instances.append({
            "video_id": video_id
        })

    if len(instances) == 0:
        print(f"[WARN] No .npy files in {class_dir}")
        continue

    annotations.append({
        "id": class_id,     # ⭐ keep numeric/string ID
        "gloss": gloss,     # ⭐ text label
        "instances": instances
    })

    total_samples += len(instances)

# -----------------------------
# Save output JSON
# -----------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2)

print(f"✅ Saved {OUTPUT_JSON}")
print(f"📊 Total glosses: {len(annotations)}")
print(f"📊 Total samples: {total_samples}")

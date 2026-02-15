import json

txt_path = r"E:\Datasets\wlasl300_dataset\labels.txt"     
json_path = r"E:\Balanced_20_Frames_Augmented\NPY\train_final.json" 

label_map = {}

with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx, label = line.split(maxsplit=1)
        label_map[idx] = label

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=2)

print(f"Saved JSON to {json_path}")

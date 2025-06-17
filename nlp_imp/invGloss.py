import json

INP = 'gloss_dataset.json'
OUT = 'inv_gloss.json'

with open(INP, 'r', encoding='utf-8') as f:
    gloss_data = json.load(f)

inv_gloss = {}
for key, values in gloss_data.items():
    inv_gloss[values] = key

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(inv_gloss, f, ensure_ascii=False, indent=2)

import json

INP = 'inv_gloss.json'
OUT = 'invGlossList.json'

with open(INP, 'r', encoding='utf-8') as f:
    gloss_data = json.load(f)

invList = {}

for key, val in gloss_data.items():
    invList[key.lower()] = [val]

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(invList, f, ensure_ascii=False, indent=2)
# 🧠 1. Structure of each `.npy` file

Each file:

```text
(T, 75, 3)
```

### 🔹 Meaning

* **T** → number of frames in the video
* **75** → total keypoints per frame
* **3** → `(x, y, confidence/visibility)`

---

## 📌 Keypoint breakdown (per frame)

```text
0 – 32   → Pose (33 points)
33 – 53  → Left Hand (21 points)
54 – 74  → Right Hand (21 points)
```

So one frame looks like:

```text
Frame_t:
[
  [x0, y0, c0],   # pose
  ...
  [x32, y32, c32],

  [x33, y33, c33],  # left hand
  ...
  [x53, y53, c53],

  [x54, y54, c54],  # right hand
  ...
  [x74, y74, c74]
]
```

---

## 🧩 What your code assumes

Inside your pipeline:

```python
pose = pts2d[0:33]
left = pts2d[33:54]
right = pts2d[54:75]
```

So:

* Full skeleton = **pose + both hands**
* No face landmarks (important)

---

# 🎥 2. Structure of the video (sequence level)

Each `.npy` = **one video sample**

```text
Video_i.npy → sequence of frames

[
  Frame_1 (75,3),
  Frame_2 (75,3),
  ...
  Frame_T (75,3)
]
```

So conceptually:

```text
Video = temporal sequence of skeletons
```

---

# 📂 3. Folder structure (WLASL-style)

Your path:

```text
E:/WLASL/wlasl_1000_preproc/videos/1/01610.npy
```

### 🔹 Interpreted structure

```text
wlasl_1000_preproc/
│
├── videos/
│   ├── 1/          ← class label (sign ID)
│   │   ├── 00001.npy
│   │   ├── 01610.npy
│   │   └── ...
│   │
│   ├── 2/
│   ├── 3/
│   └── ...
```

---

## 🏷️ Meaning

* `videos/` → dataset root
* `1/` → **class index (sign label)**
* `01610.npy` → **one sample/video of that sign**

---

# 🔄 4. Full data pipeline interpretation

```text
Dataset
  └── Class (sign word)
        └── Video sample (.npy)
              └── Frames (T)
                    └── Keypoints (75)
                          └── (x, y, confidence)
```

---

# ⚙️ 5. How your rendering works

Your code:

1. Takes all frames
2. Computes **global normalization (scale + center)**
3. Draws:

   * Pose (green)
   * Left hand (blue)
   * Right hand (red)
4. Writes video

So you're converting:

```text
Skeleton sequence → visual animation
```

---

# 🚀 6. Important observations (for your ML work)

* ✔ Temporal info = **sequence length (T)**
* ✔ Spatial info = **(x, y)**
* ✔ Confidence = optional (you currently ignore it)
* ✔ Normalization = **global per video (good choice)**

---

# ⚠️ Potential pitfalls

* Different videos → different T (variable length)
* Missing joints → low confidence values
* Scale varies across dataset → you're fixing it globally (good)



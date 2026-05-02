# LSMU Params

### LSMU Parameter Table (Extended & Practical)

| Category | Parameter | Type | Possible Values / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Hand (H)** | `handshape` | Categorical | flat-b, open-5, fist(S), index-1, C, O, L, W, A, claw, pinch, other | Overall hand configuration |
| | `finger_extension` | Vector (5) | [0–1] per finger | Thumb → pinky extension level |
| | `flexion_level` | Float | 0–1 | Overall hand openness |
| | `symmetry` | Categorical | one_hand, symmetric, asymmetric | Hand usage type |
| **Orientation (O)** | `palm_orientation` | Categorical | palm-up, down, left, right, in, out | Discrete palm direction |
| | `palm_normal_vector` | Vector (3) | [-1, 1] | 3D palm direction |
| | `wrist_rotation` | Float | Angle (° or rad) | Wrist twist |
| **Location (S)** | `location_label` | Categorical | chin, mouth, cheek, temple, forehead, chest, shoulder, torso, side, neutral | Body-relative zone |
| | `location_relative` | Vector (2/3) | Normalized Coords | Position relative to torso |
| **Movement (M)** | `movement_type` | Categorical | hold, tap, linear, arc, circular, zigzag, shake, brush, twist, approach, separate | Motion pattern |
| | `movement_dir` | Categorical | up, down, left, right, forward, backward | Direction of motion |
| | `trajectory` | Sequence | Variable | Full motion path (optional) |
| | `speed` | Float/Cat | slow, normal, fast (or continuous) | Motion speed |
| | `amplitude` | Float/Cat | small, medium, large (or continuous) | Motion size |
| | `repetition_count` | Integer | ≥1 | Number of repeats |
| **Temporal (T)** | `duration_ms` | Float | Milliseconds | Duration of this unit |
| | `hold_time` | Float | Milliseconds | Pause duration |
| | `transition_time` | Float | Milliseconds | Transition between LSMUs |
| | `phase_count` | Integer | ≥1 | Multi-stage structure |
| **Two-Hand (2H)** | `contact` | Boolean | True / False | Hands touching or not |
| | `relative_pos` | Categorical | left_above_right, parallel, crossed | Spatial relation |
| | `synchronization` | Categorical | sync, async | Timing relation |
| **Confidence (C)** | `confidence` | Float | 0–1 | Model certainty |
| **Flags (F)** | `plural` | Boolean | True / False | Repetition-based plurality |
| | `negation` | Boolean | True / False | Negative meaning |
| | `intense` | Boolean | True / False | Emphasis / strength |
| | `question` | Boolean | True / False | Interrogative |
| | `aspect` | Categorical | continuous, habitual, iterative | Temporal grammar |
| | `classifier_usage` | Boolean | True / False | Classifier involvement |


```
{
  "word": "book",
  "handshape": "fist",
  "orientation": "palm_up",
  "movement": "horizontal",
  "location": "head",
  "location_relative": [0.12, -0.31],
  "amplitude": 0.22,
  "speed": 0.08,
  "repetition_count": 2,
  "phase_count": 1,
  "symmetry": "symmetric",
  "contact": false,
  "confidence": 0.87,
  "flags": {
    "plural": false,
    "negation": false,
    "intense": false
  }
}
```
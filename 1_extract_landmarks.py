import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.1
)
detector = vision.HandLandmarker.create_from_options(options)

def normalize_landmarks(hand_landmarks):
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    if scale == 0:
        scale = 1.0
    norm = []
    for lm in hand_landmarks:
        norm.append((lm.x - min_x) / scale)
        norm.append((lm.y - min_y) / scale)
    return norm

REF_DIR = './reference_signs'
data = []
labels = []

print("Extracting normalized hand shapes from ./reference_signs ...")

for file in sorted(os.listdir(REF_DIR)):
    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    name_only, ext = os.path.splitext(file)
    label = name_only.upper()
    full_img_path = os.path.join(REF_DIR, file)

    img = cv2.imread(full_img_path)
    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            base_coords = normalize_landmarks(hand_landmarks)
            if len(base_coords) == 42:
                base_arr = np.array(base_coords)
                data.append(base_arr.tolist())
                labels.append(label)

                np.random.seed(42)
                for _ in range(39):
                    noise = np.random.normal(0, 0.015, size=42)
                    data.append((base_arr + noise).tolist())
                    labels.append(label)

                print(f"Letter '{label}': extracted 40 normalized samples")
            break

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Completed! Saved {len(data)} samples to data.pickle")





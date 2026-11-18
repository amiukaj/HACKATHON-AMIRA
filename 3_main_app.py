import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write


if not os.path.exists('./model.p'):
    raise FileNotFoundError("model.p not found! Please run python 2_train_model.py first.")

with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']


engine = pyttsx3.init()
engine.setProperty('rate', 150)


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions or index.")

current_sentence = ""
last_letter = ""
hold_counter = 0
HOLD_THRESHOLD = 15 
status_message = "Ready - Show a sign or press [V] to speak"

def translate_speech_to_signs(text):
    """Displays sign language images from ./reference_signs for each letter in text"""
    clean_text = text.upper()
    for idx, char in enumerate(clean_text):
        if char == ' ':
            blank = np.zeros((420, 420, 3), dtype=np.uint8)
            cv2.putText(blank, "[ SPACE ]", (110, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
            cv2.imshow("Speech to Sign Translation", blank)
            cv2.waitKey(500)
        elif 'A' <= char <= 'Z':
            img_path = os.path.join('./reference_signs', f"{char}.jpg")
            if os.path.exists(img_path):
                sign_img = cv2.imread(img_path)
                sign_img = cv2.resize(sign_img, (420, 420))
               
                cv2.rectangle(sign_img, (0, 0), (420, 75), (0, 0, 0), -1)
                cv2.putText(sign_img, f"Word: {clean_text}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(sign_img, f"Signing Letter: {char}", (15, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                cv2.imshow("Speech to Sign Translation", sign_img)
                cv2.waitKey(850)
    if cv2.getWindowProperty("Speech to Sign Translation", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow("Speech to Sign Translation")

print("=== Bidirectional Sign Language Translator Running ===")
print("[S] Speak Sentence | [V] Voice-to-Sign | [SPACE] Space | [BACKSPACE] Undo | [C] Clear | [Q] Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

  
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)

   
    cv2.rectangle(frame, (0, 0), (w, 60), (25, 25, 25), -1)
    cv2.putText(frame, status_message, (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            xs = [lm.x for lm in hand_landmarks]
            ys = [lm.y for lm in hand_landmarks]
            min_x, min_y = min(xs), min(ys)
            max_x, max_y = max(xs), max(ys)
            scale = max(max_x - min_x, max_y - min_y)
            if scale == 0:
                scale = 1.0

            data_aux = []
            for landmark in hand_landmarks:
                data_aux.append((landmark.x - min_x) / scale)
                data_aux.append((landmark.y - min_y) / scale)
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            if len(data_aux) == 42:
                prediction = model.predict(np.asarray(data_aux).reshape(1, -1))
                for pred_letter in prediction:
                    letter = str(pred_letter)

                    if letter == last_letter:
                        hold_counter += 1
                        if hold_counter >= HOLD_THRESHOLD:
                            current_sentence += letter
                            hold_counter = 0
                            status_message = f"Added letter: {letter}"
                    else:
                        last_letter = letter
                        hold_counter = 0

                    cv2.putText(frame, f"Sign: {letter}", (20, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                    bar_width = int((hold_counter / HOLD_THRESHOLD) * 150)
                    cv2.rectangle(frame, (20, 120), (170, 135), (80, 80, 80), -1)
                    cv2.rectangle(frame, (20, 120), (20 + bar_width, 135), (0, 255, 0), -1)
            break
    else:
        hold_counter = 0

    
    cv2.rectangle(frame, (0, h - 90), (w, h), (15, 15, 15), -1)
    cv2.putText(frame, f"Output: {current_sentence}", (15, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "[S] Speak  [V] Voice->Sign  [Space] _  [Bksp] Undo  [C] Clear  [Q] Quit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow('Bidirectional Sign Language Translator', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        current_sentence += " "
        status_message = "Added space"
    elif key == 8: 
        current_sentence = current_sentence[:-1]
        status_message = "Deleted last character"
    elif key == ord('c'):
        current_sentence = ""
        status_message = "Cleared output"
    elif key == ord('s') and current_sentence.strip() != "":
        status_message = f"Speaking: {current_sentence}"
        cv2.imshow('Bidirectional Sign Language Translator', frame)
        cv2.waitKey(1)
        engine.say(current_sentence)
        engine.runAndWait()
        status_message = "Ready - Show a sign or press [V] to speak"
    elif key == ord('v'):
      
        status_message = "LISTENING (Speak for 4 seconds)..."
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 180), -1)
        cv2.putText(frame, status_message, (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imshow('Bidirectional Sign Language Translator', frame)
        cv2.waitKey(1)

        r = sr.Recognizer()
        fs = 44100
        seconds = 4
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        write('temp_audio.wav', fs, recording)

        with sr.AudioFile('temp_audio.wav') as source:
            audio = r.record(source)
            try:
                spoken_text = r.recognize_google(audio)
                status_message = f"Heard: '{spoken_text}' -> Translating to signs..."
                translate_speech_to_signs(spoken_text)
                status_message = f"Translated: '{spoken_text}'"
            except Exception:
                status_message = "Could not understand voice - Press [V] to try again"

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque
from textblob import TextBlob
import requests


API_KEY = "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"

# --- Mediapipe Hands setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --- Load your trained model ---
model_load_path = 'knn_asl_model.pkl'
try:
    knn_model = joblib.load(model_load_path)
    print(f"Loaded KNN model from {model_load_path}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

label_map = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

buffer_time = 5  # seconds
prediction_buffer = deque()
last_capture_time = time.time()
collected_letters = []

cap = cv2.VideoCapture(0)
print("Starting webcam... Press 'q' to quit.")

# --- FPS calculation ---
prev_frame_time = time.time()
fps = 0

def handle_gesture(letter, collected_letters):
    """Handle special gestures for del and space."""
    if letter == 'del' and collected_letters:
        collected_letters.pop()
    elif letter == 'space':
        collected_letters.append(' ')
    elif letter not in ['nothing', 'del', 'space']:
        collected_letters.append(letter)
    return collected_letters

def get_small_talk_reply(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant for an academic vision system project. "
                    "Only answer questions or prompts related to sign language, computer vision, or this system's output. "
                    "If the prompt is unrelated, politely refuse."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error getting reply: {e}")
        return "❌ Error getting reply."

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    predicted_sign = "No Hand Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if knn_model:
                try:
                    prediction = knn_model.predict(np.array(landmarks).reshape(1, -1))[0]
                    if isinstance(prediction, (int, np.integer)):
                        if 0 <= prediction < len(label_map):
                            predicted_sign = label_map[prediction]
                        else:
                            predicted_sign = "Prediction Index Out of Bounds"
                    elif isinstance(prediction, str):
                        predicted_sign = prediction
                    else:
                        predicted_sign = f"Unknown prediction type: {type(prediction)}"
                except Exception as e:
                    print(f"Prediction error: {e}")
                    predicted_sign = "Prediction Error"
            else:
                predicted_sign = "Model not loaded"

    # Show predicted sign on frame
    cv2.putText(frame, f"Predicted: {predicted_sign}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Buffer predictions for autocorrect every 5 seconds
    current_time = time.time()
    if predicted_sign not in ["No Hand Detected", "Prediction Error", "Model not loaded", "Prediction Index Out of Bounds"]:
        prediction_buffer.append(predicted_sign)

    # Every 5 seconds, select the most frequent predicted letter in the buffer
    if (current_time - last_capture_time) > buffer_time:
        if prediction_buffer:
            most_common_letter = max(set(prediction_buffer), key=prediction_buffer.count)
            collected_letters = handle_gesture(most_common_letter, collected_letters)
            print(f"Selected letter after {buffer_time}s: {most_common_letter}")
            prediction_buffer.clear()
            last_capture_time = current_time

        # Autocorrect current sentence
        sentence = "".join(collected_letters)
        corrected_sentence = str(TextBlob(sentence).correct())
        print(f"Sentence so far: {sentence}")
        print(f"Autocorrected: {corrected_sentence}")

    # Show current sentence on frame (autocorrected)
    sentence = "".join(collected_letters)
    corrected_sentence = str(TextBlob(sentence).correct())
    cv2.putText(frame, f"Sentence: {corrected_sentence}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # --- FPS calculation and display ---
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("ASL Prediction Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    # --- New Features ---
    if key == ord('q'):
        break
    elif key == ord('s'):
        if corrected_sentence.strip():
            print("Sending to DeepSeek/OpenRouter...")
            reply = get_small_talk_reply(corrected_sentence)
            print(f"DeepSeek Reply: {reply}")
    elif key == 32:  # Spacebar
        collected_letters.append(' ')
    elif key == 8:   # Backspace
        if collected_letters:
            collected_letters.pop()

cap.release()
cv2.destroyAllWindows()

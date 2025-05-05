import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
import sys
from pynput.keyboard import Key, Controller

# ตั้งค่า working directory
os.chdir('/Users/aoyrzz/Desktop/hand_game/main_project')
print("Current working directory:", os.getcwd())

# เพิ่ม path เพื่อให้ Python ค้นหาโมดูล
sys.path.append(os.getcwd())

# นำเข้าฟังก์ชันจาก Ex_feature.py
try:
    from Extrack_feature.Ex_feature import calculate_distances_and_angles
except ModuleNotFoundError as e:
    print("Error: Could not import calculate_distances_and_angles. Check if Extrack_feature/Ex_feature.py exists")
    print("Current sys.path:", sys.path)
    raise e

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7  # เพิ่มเพื่อความเสถียร
)

# โหลดโมเดล
model_file = 'Train/model/rf_model.pkl'
if not os.path.exists(model_file):
    print(f"Error: {model_file} not found")
    exit()
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# ตั้งค่า keyboard controller
keyboard = Controller()

# เริ่มกล้อง
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# ตัวแปรสำหรับควบคุมการกดปุ่ม
last_gesture = None
key_press_duration = 0.1  # ระยะเวลากดปุ่ม (วินาที)

# วนลูปตรวจจับมือและควบคุมเกม
previous_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_name = "Unknown"
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # ดึงพิกัด landmark และแปลงเป็น NumPy array
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks)  # ขนาด (63,)
        print("Landmarks shape:", landmarks.shape)

        # คำนวณฟีเจอร์
        features = calculate_distances_and_angles(landmarks)
        print("Features length:", len(features))  # ควรได้ 17

        # แปลงฟีเจอร์เป็น NumPy array และ reshape
        features_reshaped = np.array(features).reshape(1, -1)
        print("Features reshaped shape:", features_reshaped.shape)  # ควรได้ (1, 17)

        # ทำนายท่าทาง
        gesture_name = model.predict(features_reshaped)[0]

        # จำลองการกดปุ่มตามท่าทาง
        if gesture_name != last_gesture:
            if gesture_name == "Up":  # กระโดด
                keyboard.press(Key.up)
                time.sleep(key_press_duration)
                keyboard.release(Key.up)
            elif gesture_name == "Down":  # หมอบ
                keyboard.press(Key.down)
                time.sleep(key_press_duration)
                keyboard.release(Key.down)
            elif gesture_name == "Left":  # เลื่อนซ้าย
                keyboard.press(Key.left)
                time.sleep(key_press_duration)
                keyboard.release(Key.left)
            elif gesture_name == "Right":  # เลื่อนขวา
                keyboard.press(Key.right)
                time.sleep(key_press_duration)
                keyboard.release(Key.right)
            last_gesture = gesture_name

        # วาด landmarks บนภาพ
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # แสดง FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # แสดงข้อความ
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Subway Surfers Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้อง
cap.release()
cv2.destroyAllWindows()
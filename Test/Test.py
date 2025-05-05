# ส่วนที่ 1: นำเข้าไลบรารีและโหลดโมเดล
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import time
from itertools import combinations

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.5)

# โหลดโมเดล (เลือก Random Forest เป็นหลัก)
model_file = 'Train/model2/rf_model.pkl'  # สามารถเปลี่ยนเป็น 'knn_model.pkl' หรือ 'svm_model.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# เริ่มกล้อง
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# -----------------------------------------------

# ส่วนที่ 2: ฟังก์ชันคำนวณฟีเจอร์ (เหมือนใน extract_features.py)
# คำนวณระยะห่างและมุมระหว่าง landmarks
key_points = [0, 4, 8, 12, 16, 20]  # wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip
landmarks_indices = []
for i in range(21):
    landmarks_indices.extend([i*3, i*3+1, i*3+2])  # x, y, z สำหรับแต่ละ landmark

# สร้างรายการคู่ของ key points สำหรับคำนวณระยะห่าง
distance_pairs = list(combinations(key_points, 2))

x_coords = np.zeros(21)
y_coords = np.zeros(21)
z_coords = np.zeros(21)

features = np.zeros(len(distance_pairs) + 2)

# -----------------------------------------------

# ส่วนที่ 3: ลูปเรียลไทม์สำหรับทดสอบ
previous_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # กลับภาพเพื่อความถูกต้อง
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_name = "Unknown"
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # ดึงพิกัด x, y, z
        landmarks = []
        for i, lm in enumerate(hand_landmarks.landmark):
            x_coords[i] = lm.x
            y_coords[i] = lm.y
            z_coords[i] = lm.z
            landmarks.extend([lm.x, lm.y, lm.z])

        # คำนวณระยะห่าง
        for idx, (i, j) in enumerate(distance_pairs):
            dist = np.sqrt((x_coords[i] - x_coords[j])**2 + 
                          (y_coords[i] - y_coords[j])**2 + 
                          (z_coords[i] - z_coords[j])**2)
            features[idx] = dist

        # คำนวณมุม
        def calculate_angle(p1, p2, p3):
            v1 = np.array([x_coords[p1] - x_coords[p2], y_coords[p1] - y_coords[p2], z_coords[p1] - z_coords[p2]])
            v2 = np.array([x_coords[p3] - x_coords[p2], y_coords[p3] - y_coords[p2], z_coords[p3] - z_coords[p2]])
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)

        angle1 = calculate_angle(8, 0, 12)  # index_tip, wrist, middle_tip
        angle2 = calculate_angle(4, 0, 8)   # thumb_tip, wrist, index_tip
        features[-2] = angle1
        features[-1] = angle2

        # ทำนายท่าทาง
        features_reshaped = features.reshape(1, -1)
        gesture_name = model.predict(features_reshaped)[0]

        # วาด landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # คำนวณและแสดง FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # แสดงผลบนหน้าจอ
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Realtime Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
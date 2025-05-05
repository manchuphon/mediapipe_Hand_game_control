import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from Extrack_feature.Ex_feature import calculate_distances_and_angles   # ✅ ใช้ฟังก์ชันที่คุณเขียนไว้

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# โหลดโมเดล
model_file = '/Users/aoyrzz/Desktop/hand_game/main_project/Train/model2/rf_model.pkl'  # เปลี่ยนตามชื่อไฟล์ของคุณ
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# เริ่มกล้อง
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# วนลูปตรวจจับมือและทำนาย
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

        # ดึงพิกัด landmark และเรียกฟังก์ชัน extract feature
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])  # รวมเป็น (63,)

        # คำนวณฟีเจอร์
        features = calculate_distances_and_angles(landmarks)
        features_reshaped = np.array(features).reshape(1, -1)

        # ทำนายท่าทาง
        gesture_name = model.predict(features_reshaped)[0]

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

    cv2.imshow("Realtime Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้อง
cap.release()
cv2.destroyAllWindows()

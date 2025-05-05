import cv2
import mediapipe as mp
import pandas as pd

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

# รับ label จากผู้ใช้
label = input("พิมพ์ชื่อท่าทางที่ต้องการบันทึก (label): ").strip()

data = []

columns = []
for i in range(21):
    columns += [f"x{i}_right", f"y{i}_right", f"z{i}_right"]
columns.append("label")

print("กด 's' เพื่อบันทึกข้อมูล และ กด 'q' เพื่อออก")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    keypoint_count = 0
    row = []

    display_frame = cv2.flip(frame, 1)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoint_count = len(hand_landmarks.landmark)

        h, w, _ = display_frame.shape
        flipped_landmarks = []
        for lm in hand_landmarks.landmark:
            x = int((1 - lm.x) * w)
            y = int(lm.y * h)
            flipped_landmarks.append((x, y))
            cv2.circle(display_frame, (x, y), 5, (255, 0, 0), -1)

        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = flipped_landmarks[start_idx]
            end_point = flipped_landmarks[end_idx]
            cv2.line(display_frame, start_point, end_point, (0, 255, 0), 2)

    cv2.putText(display_frame, f"Data Count: {len(data)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Keypoints detected: {keypoint_count}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow('Collecting Hand Data', display_frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('s'):
        if keypoint_count == 21:
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            row.append(label)
            data.append(row)
            print(f"บันทึกข้อมูลชุดที่ {len(data)} แล้ว สำหรับ label: {label}")
        else:
            print("ไม่พบ keypoints กรุณาแสดงมือก่อนกด 's'")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# บันทึกทั้งหมดลงไฟล์เดียว
csv_file = "gestures.csv"

try:
    df_existing = pd.read_csv(csv_file)
    df_new = pd.DataFrame(data, columns=columns)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(csv_file, index=False)
except FileNotFoundError:
    df_new = pd.DataFrame(data, columns=columns)
    df_new.to_csv(csv_file, index=False)

print(f"บันทึกข้อมูลทั้งหมด {len(data)} แถว ลงใน {csv_file}")

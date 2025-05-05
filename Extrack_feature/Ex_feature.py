import pandas as pd
import numpy as np
from itertools import combinations

def calculate_distances_and_angles(landmarks):
    """คำนวณระยะห่างและมุมระหว่าง landmarks"""
    # landmarks เป็น array ขนาด (63,) คิดจาก (x, y, z) ของ 21 จุด
    features = []
    
    # แยก x, y, z
    x = landmarks[::3]  # x0, x1, ..., x20
    y = landmarks[1::3] # y0, y1, ..., y20
    z = landmarks[2::3] # z0, z1, ..., z20
    
    # คำนวณระยะห่าง Euclidean ระหว่างคู่ของ landmarks สำคัญ
    key_points = [0, 4, 8, 12, 16, 20]  # wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip
    for (i, j) in combinations(key_points, 2):
        dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)
        features.append(dist)
    
    # คำนวณมุมระหว่างคู่ของนิ้ว (ใช้ index_tip, middle_tip, wrist)
    def calculate_angle(p1, p2, p3):
        """คำนวณมุมระหว่าง 3 จุด (p2 เป็นจุดยอด)"""
        v1 = np.array([x[p1] - x[p2], y[p1] - y[p2], z[p1] - z[p2]])
        v2 = np.array([x[p3] - x[p2], y[p3] - y[p2], z[p3] - z[p2]])
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    # มุมระหว่างนิ้วชี้-ข้อมือ-นิ้วกลาง
    angle1 = calculate_angle(8, 0, 12)  # index_tip, wrist, middle_tip
    # มุมระหว่างนิ้วโป้ง-ข้อมือ-นิ้วชี้
    angle2 = calculate_angle(4, 0, 8)   # thumb_tip, wrist, index_tip
    features.extend([angle1, angle2])
    
    return features

def extract_features(input_csv, output_csv):
    """ดึงฟีเจอร์จาก gestures.csv และบันทึกเป็น gestures_features.csv"""
    # อ่านข้อมูล
    df = pd.read_csv(input_csv)
    labels = df['label'].values
    landmarks = df.drop('label', axis=1).values  # (x0_right, y0_right, z0_right, ...)
    
    # ดึงฟีเจอร์
    features = []
    for lm in landmarks:
        feat = calculate_distances_and_angles(lm)
        features.append(feat)
    
    # สร้าง DataFrame
    feature_columns = [f'dist_{i}' for i in range(len(features[0]) - 2)] + ['angle_1', 'angle_2']
    df_features = pd.DataFrame(features, columns=feature_columns)
    df_features['label'] = labels
    
    # บันทึกเป็น CSV
    df_features.to_csv(output_csv, index=False)
    print(f"บันทึกฟีเจอร์ลง {output_csv}, จำนวนแถว: {len(df_features)}")

if __name__ == "__main__":
    input_csv = "gestures.csv"
    output_csv = "gestures_features.csv"
    extract_features(input_csv, output_csv)
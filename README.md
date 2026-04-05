# 🖐️ Hand Game Control

> ควบคุมเกมด้วยท่าทางมือแบบ Real-time ผ่านกล้องเว็บแคม  
> โดยใช้ **MediaPipe** + **Machine Learning** จำแนกท่าทางมือเป็นคำสั่งควบคุมเกม

---

## 📌 ภาพรวมโปรเจกต์

โปรเจกต์นี้สร้างระบบควบคุมเกม (เช่น **Subway Surfers**) ด้วยท่าทางมือ โดยไม่ต้องใช้ keyboard หรือ controller  
กล้องเว็บแคมจะตรวจจับมือแบบ real-time → สกัด features → ส่งเข้า ML model → แปลงผลลัพธ์เป็น arrow key

**Gesture ที่รองรับ:**

| ท่าทาง | คีย์ที่ส่ง | การกระทำในเกม |
|--------|-----------|---------------|
| ✋ Up   | ↑ Arrow Up | กระโดด |
| 👇 Down | ↓ Arrow Down | หมอบ |
| 👈 Left | ← Arrow Left | เลื่อนซ้าย |
| 👉 Right | → Arrow Right | เลื่อนขวา |

---

## 🗂️ โครงสร้างโปรเจกต์

```
Hand_game_control/
│
├── data/
│   ├── co_data.py              # เก็บ dataset ผ่านกล้องเว็บแคม
│   └── gestures.csv            # ข้อมูล landmark 21 จุดของมือ (63 features + label)
│
├── Extrack_feature/
│   ├── Ex_feature.py           # สกัด features จาก landmarks (ระยะห่าง + มุม)
│   └── gestures_features.csv   # Feature dataset ที่พร้อมสำหรับ training (17 features)
│
├── Train/
│   ├── train_models.ipynb      # Training + evaluation โมเดล v1 (model/)
│   ├── trainmodel2.ipynb       # Training + evaluation โมเดล v2 (model2/)
│   ├── model/
│   │   ├── rf_model.pkl        # Random Forest v1
│   │   ├── knn_model.pkl       # KNN v1
│   │   └── svm_model.pkl       # SVM v1
│   ├── model2/
│   │   ├── rf_model.pkl        # Random Forest v2
│   │   ├── knn_model.pkl       # KNN v2
│   │   └── svm_model.pkl       # SVM v2
│   └── model_performance_comparison.png
│
├── Test/
│   ├── Test.py                 # ทดสอบ real-time gesture recognition (model2, inline features)
│   ├── Test2.py                # ทดสอบโดย import Ex_feature (model2)
│   └── Test3.py                # ทดสอบโดย import Ex_feature (model v1, พร้อม debug log)
│
├── play/
│   └── playgame.py             # รันระบบควบคุมเกมจริง (ส่ง arrow key ผ่าน pynput)
│
└── image/
    ├── accuracy_comparison.png      # กราฟเปรียบเทียบ accuracy
    ├── confusion_matrices.png       # Confusion matrix ทั้ง 3 โมเดล
    ├── feature_importance_rf.png    # Feature importance (Random Forest)
    ├── learning_curves.png          # Learning curves
    ├── per_class_metrics.png        # Precision / Recall / F1 แยกตาม class
    ├── roc_curve_rf.png             # ROC curve (Random Forest)
    └── roc_curves.png               # ROC curves เปรียบเทียบ
```

---

## ⚙️ Pipeline การทำงาน

```
[กล้อง] → [MediaPipe Hand Landmark] → [Feature Extraction] → [ML Model] → [Keyboard Input] → [เกม]
```

### 1️⃣ เก็บ Dataset — `data/co_data.py`

ใช้ MediaPipe ตรวจจับ 21 landmarks ของมือ (x, y, z) = 63 ค่าต่อ frame  
กด `s` เพื่อบันทึกท่าทาง, กด `q` เพื่อออก → บันทึกรวมใน `gestures.csv`

```
คอลัมน์: x0_right, y0_right, z0_right, ..., x20_right, y20_right, z20_right, label
```

### 2️⃣ สกัด Features — `Extrack_feature/Ex_feature.py`

แปลง raw landmarks (63 ค่า) เป็น geometric features 17 ค่า:

- **15 ระยะห่าง Euclidean** ระหว่าง 6 key points (wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip) — จาก C(6,2) = 15 คู่
- **2 มุม (Angle):**
  - `angle_1`: index_tip → wrist → middle_tip
  - `angle_2`: thumb_tip → wrist → index_tip

ผลลัพธ์บันทึกใน `gestures_features.csv`

### 3️⃣ Training โมเดล — `Train/train_models.ipynb`

ฝึก 3 โมเดลพร้อมกัน พร้อม evaluation แบบครบถ้วน:

| โมเดล | Parameter |
|-------|-----------|
| Random Forest | `n_estimators=100` |
| KNN | `n_neighbors=5` |
| SVM | `kernel='rbf'` |

**Evaluation ที่ทำ:**
- Accuracy Comparison (bar plot)
- Confusion Matrix (ทั้ง 3 โมเดล)
- Precision / Recall / F1-Score แยกตาม class
- ROC Curve + AUC
- Feature Importance (Random Forest)
- Learning Curves

### 4️⃣ ทดสอบ Real-time — `Test/`

| ไฟล์ | รายละเอียด |
|------|-----------|
| `Test.py` | ใช้ model2, inline feature calculation |
| `Test2.py` | ใช้ model2, import `calculate_distances_and_angles` จาก `Ex_feature.py` |
| `Test3.py` | ใช้ model v1, มี debug output (shape, length) |

### 5️⃣ ควบคุมเกมจริง — `play/playgame.py`

โหลด `Train/model/rf_model.pkl` → ทำนาย gesture → ส่ง arrow key ผ่าน `pynput`  
แสดง gesture name และ FPS บน frame แบบ real-time

---

## 🛠️ การติดตั้ง

### ความต้องการของระบบ

- Python 3.8+
- เว็บแคม
- macOS / Windows / Linux

### ติดตั้ง Dependencies

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn pynput seaborn matplotlib
```

---

## 🚀 วิธีใช้งาน

### ขั้นตอนที่ 1: เก็บข้อมูล

```bash
cd data
python co_data.py
# พิมพ์ชื่อท่าทาง เช่น: Up, Down, Left, Right
# กด 's' เพื่อบันทึก, กด 'q' เพื่อออก
```

### ขั้นตอนที่ 2: สกัด Features

```bash
cd Extrack_feature
python Ex_feature.py
```

### ขั้นตอนที่ 3: Train โมเดล

เปิด `Train/train_models.ipynb` ใน Jupyter Notebook แล้วรัน cell ทั้งหมด

### ขั้นตอนที่ 4: ทดสอบระบบ

```bash
python Test/Test2.py
```

### ขั้นตอนที่ 5: เล่นเกม

เปิดเกมก่อน จากนั้นรัน:

```bash
python play/playgame.py
```

> ⚠️ แก้ไข path ใน `playgame.py` บรรทัด `os.chdir(...)` ให้ตรงกับที่ตั้งโปรเจกต์ของคุณก่อนใช้งาน

---

## 📊 ผลการทดสอบโมเดล

กราฟ evaluation ทั้งหมดอยู่ในโฟลเดอร์ `image/`:

| ไฟล์ | เนื้อหา |
|------|---------|
| `accuracy_comparison.png` | เปรียบเทียบ accuracy ของทั้ง 3 โมเดล |
| `confusion_matrices.png` | Confusion matrix แยกตามโมเดล |
| `per_class_metrics.png` | Precision / Recall / F1 แยกตาม gesture class |
| `roc_curves.png` | ROC curves ทั้ง 3 โมเดล |
| `roc_curve_rf.png` | ROC curve เฉพาะ Random Forest |
| `feature_importance_rf.png` | Feature สำคัญที่สุดในการแยก gesture |
| `learning_curves.png` | การเรียนรู้ของโมเดลตามขนาด dataset |

---

## 🧰 Tech Stack

| ส่วน | เทคโนโลยี |
|------|-----------|
| Hand Tracking | [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) |
| Computer Vision | OpenCV |
| Machine Learning | scikit-learn (Random Forest, KNN, SVM) |
| Feature Engineering | NumPy, pandas |
| Keyboard Control | pynput |
| Visualization | matplotlib, seaborn |
| Notebook | Jupyter Notebook |

---

## 👤 ผู้พัฒนา

**Manchuphon (Aoy)**  


---

## 📄 License

This project is for educational purposes.

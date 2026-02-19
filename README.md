# 🖐️ AI Gesture Control System  
### Hand Tracking + Machine Learning + Real-Time System Control

An end-to-end **Gesture Recognition & Control System** built using:

- OpenCV  
- MediaPipe Hand Landmarker  
- Scikit-Learn (Random Forest)  
- FastAPI  
- PyAutoGUI  

This system allows you to:

✔ Collect custom gesture datasets  
✔ Train your own ML model  
✔ Control mouse & keyboard using gestures  
✔ Configure gesture mappings from a web interface  

---

# 🚀 Complete Workflow (Single Overview)

Step 1 → Collect Data (take_data.py)
Step 2 → Train Model (train.py)
Step 3 → Run Real-Time App (backend.py)


---

# 📂 Project Structure

gesture-control/
│
├── take_data.py
├── train.py
├── backend.py
├── gesture_dataset.csv
├── gesture_model.pkl
├── gestures.json
├── hand_landmarker.task
├── templates/
└── static/

# 🧠 How The System Works

### 1️⃣ Hand Detection
- MediaPipe detects 21 hand landmarks.
- Each landmark provides (x, y, z) coordinates.

### 2️⃣ Feature Engineering
- Landmarks are normalized relative to wrist (landmark 0).
- Scale normalization applied.
- Total features = 63 per sample.

### 3️⃣ Model Training
- RandomForestClassifier trained on dataset.
- Dataset split 80/20.
- Accuracy printed.
- Model saved as `gesture_model.pkl`.

### 4️⃣ Real-Time Prediction
- Webcam frame captured.
- Features extracted.
- Model predicts gesture probabilities.
- If confidence ≥ threshold → mapped system action executed.

---

# 🎮 Real-Time Features

✔ Smooth mouse movement via index finger  
✔ Pinch detection for left-click  
✔ Confidence threshold filtering  
✔ Cooldown mechanism to prevent spamming  
✔ Web-based gesture mapping  
✔ Supports keyboard, hotkeys, URLs, and apps  

---

# 👨‍💻 Author
Sarthak Tomar

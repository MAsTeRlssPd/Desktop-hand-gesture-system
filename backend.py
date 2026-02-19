import cv2
import mediapipe as mp
import pyautogui
import json
import threading
import time
import pickle
import os
import numpy as np
import subprocess
import webbrowser
import math
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import uvicorn 
import warnings
warnings.filterwarnings("ignore")  
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()
smooth_x, smooth_y = 0, 0
smoothing_factor = 0.1 
LEFT_CLICK_THRESHOLD = 0.04 
CLICK_COOLDOWN = 2        
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
GESTURE_CONFIG = "gestures.json"
MODEL_FILE = "gesture_model.pkl"
COOLDOWN_DURATION = 3.0 
CONFIDENCE_THRESHOLD = 0.95 
gesture_mappings = {"open_palm": "space", "fist": "esc", "peace": "volumemute"}

def load_config():
    global gesture_mappings
    if os.path.exists(GESTURE_CONFIG):
        try:
            with open(GESTURE_CONFIG, "r") as f:
                gesture_mappings = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
try:
    with open(MODEL_FILE, 'rb') as f:
        gesture_model = pickle.load(f)
except FileNotFoundError:
    gesture_model = None

def extract_features(landmarks):
    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    features = []
    for lm in landmarks:
        features.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    max_val = max(abs(v) for v in features) if max(abs(v) for v in features) != 0 else 1
    return [f / max_val for f in features]

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def vision_loop():
    global smooth_x, smooth_y
    if not gesture_model: return
    
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        running_mode=vision.RunningMode.VIDEO, 
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)
    last_action_time = 0  
    last_click_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect_for_video(mp_image, int(time.time() * 1000))
        current_time = time.time()
        display_score = 0.0
        display_color = (255, 255, 255)
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            idx_tip = landmarks[8]
            target_x, target_y = idx_tip.x * screen_w, idx_tip.y * screen_h
            smooth_x += (target_x - smooth_x) * smoothing_factor
            smooth_y += (target_y - smooth_y) * smoothing_factor
            pyautogui.moveTo(int(smooth_x), int(smooth_y), _pause=False)
            idx_down = landmarks[6]
            dist = get_distance(idx_down, idx_tip)
            if dist < LEFT_CLICK_THRESHOLD and (current_time - last_click_time) > CLICK_COOLDOWN:
                pyautogui.click()
                last_click_time = current_time
                display_color = (0, 255, 255)
            features = extract_features(landmarks)
            probabilities = gesture_model.predict_proba([features])[0]
            max_idx = np.argmax(probabilities)
            confidence = probabilities[max_idx]
            gesture = gesture_model.classes_[max_idx]
            display_score = round(confidence, 2)
            if (current_time - last_action_time) >= COOLDOWN_DURATION:
                if confidence >= CONFIDENCE_THRESHOLD:
                    action = gesture_mappings.get(gesture)
                    if action:
                        if action.startswith("http"): webbrowser.open(action)
                        elif action.startswith("open:"):
                            app_name = action.split(":", 1)[1]
                            subprocess.Popen(f"start {app_name}:", shell=True)
                        elif "+" in action:
                            pyautogui.hotkey(*action.split("+"))
                        else:
                            pyautogui.press(action)
                        last_action_time = current_time
                        display_color = (0, 255, 0)
                else:
                    display_color = (0, 165, 255)
        cv2.putText(frame, f"Conf: {display_score}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)
        cv2.imshow('Gesture Control', frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release()
    cv2.destroyAllWindows()
@app.on_event("startup")
def startup():
    load_config() 
    threading.Thread(target=vision_loop, daemon=True).start()
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "mappings": gesture_mappings})
@app.post("/update")
async def update(gesture: str, action: str):
    global gesture_mappings
    gesture_mappings[gesture] = action
    with open(GESTURE_CONFIG, "w") as f:
        json.dump(gesture_mappings, f)
    return {"status": "ok"}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
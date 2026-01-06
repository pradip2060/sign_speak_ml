import os
import cv2
import shutil
import tempfile
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
# CORS設定用のインポートを追加
from fastapi.middleware.cors import CORSMiddleware
from predict_gesture import extract_landmark_vec, CLASSES, T, LAND_DIM, MODEL_PATH
from asl_config import LABEL_MAP

app = FastAPI()

# ==============================
# CORS 設定を追加
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
async def index():
    return {"message": "Hello, this is the gesture prediction API."}

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    # 1. 一時ファイルとして保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # 2. 動画からLandmarkシーケンスを抽出
    cap = cv2.VideoCapture(tmp_path)
    all_landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        vec = extract_landmark_vec(frame)
        all_landmarks.append(vec)
    cap.release()
    
    # 一時ファイルを削除
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # シーケンス長をTに調整（短い場合はゼロパディング）
    if len(all_landmarks) < T:
        pad_len = T - len(all_landmarks)
        all_landmarks.extend([np.zeros(LAND_DIM)] * pad_len)
    
    # 3. 推論
    seq = np.array(all_landmarks[:T], dtype=np.float32).reshape(1, T, LAND_DIM)
    prediction = model.predict(seq, verbose=0)[0]
    
    # 確率最大のクラスを取得
    idx = int(np.argmax(prediction))
    prob = float(prediction[idx])

    # ラベルを取得
    key = CLASSES[idx]
    label = LABEL_MAP[key]
    return {"label": label, "probability": prob}

if __name__ == "__main__":
    import uvicorn
    # 本番環境では 0.0.0.0 で全インターフェースをリッスン
    uvicorn.run(app, host="0.0.0.0", port=8000)
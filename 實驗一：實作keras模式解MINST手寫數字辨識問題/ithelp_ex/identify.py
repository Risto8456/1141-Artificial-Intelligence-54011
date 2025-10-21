import os
import numpy as np
from keras.models import model_from_json
import cv2

# === 路徑設定 ===
base_dir = os.path.dirname(os.path.abspath(__file__))       # 目前資料夾
parent_dir = os.path.dirname(base_dir)                      # 上一層資料夾
json_path = os.path.join(base_dir, "model.config")          # 模型結構
weights_path = os.path.join(base_dir, "model.weights.h5")   # 模型權重
student_dir = os.path.join(parent_dir, "student_id_digits") # 學號各位數影像的資料夾

# === 讀取模型 ===
with open(json_path, "r") as file:
    loaded_model_json = file.read()
model = model_from_json(loaded_model_json)
model.load_weights(weights_path)
print("[Info] Model loaded successfully!\n")

# === 讀取 student_dir 內所有 .png 圖片並辨識 ===
png_files = sorted([f for f in os.listdir(student_dir) if f.lower().endswith(".png")],
                   key=lambda x: int(os.path.splitext(x)[0]))  # 按檔名數字排序

results = []  # 存每個結果 (fname, 預測數字, 信心值)

print("----- 辨識結果 -----")

for fname in png_files:
    img_path = os.path.join(student_dir, fname)
    
    # 用 np.fromfile 支援中文路徑，再用 imdecode 轉換
    img_data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"[Warning] 無法讀取圖片: {img_path}")
        continue
    
    # 縮放至 28x28、反白、正規化、攤平
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28*28)

    # 預測
    probs = model.predict(img, verbose=0)[0]
    pred_label = np.argmax(probs)
    confidence = float(probs[pred_label])

    print(f"{fname:<20} -> {pred_label}    (confidence {confidence:.3f})")

    results.append((fname, pred_label, confidence))

# === 按檔名排序後輸出完整學號 ===
sorted_digits = [str(r[1]) for r in results]
student_id = "".join(sorted_digits)

print(f"\n預測的學號（依檔名排序）: {student_id}")

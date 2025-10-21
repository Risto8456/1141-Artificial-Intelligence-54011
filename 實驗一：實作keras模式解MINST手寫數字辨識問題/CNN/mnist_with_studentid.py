# mnist_with_studentid.py
# pip install tensorflow keras pillow numpy
import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical

# ----------------------
# Config 配置
# ----------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # 目前 py 檔所在資料夾
parent_dir = os.path.dirname(current_dir)                 # 上一層資料夾

MODEL_PATH = os.path.join(current_dir, "mnist_fc_model.h5")      # 訓練後模型檔
STUDENT_DIR = os.path.join(parent_dir, "student_id_digits")      # 放學號各位數影像的資料夾
IMG_SIZE = (28, 28)
NUM_CLASSES = 10
EPOCHS = 6
BATCH_SIZE = 64

# ----------------------
# 1) 讀取並預處理 MNIST
# ----------------------
# 載入 MNIST 資料庫的訓練資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 對密集網絡進行規範化和重塑（使用小型 CNN 來獲得更好的準確率）
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 添加灰階通道
x_train = np.expand_dims(x_train, -1)  # shape (n,28,28,1)
x_test = np.expand_dims(x_test, -1)

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# ----------------------
# 2) 建造一個小型 CNN（對手繪數字更穩健）
# ----------------------
def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# If model file exists, load it; otherwise train
if os.path.exists(MODEL_PATH):
    print(f"Loading existing model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
else:
    print("Training model on MNIST ...")
    model = build_model()
    model.fit(x_train, y_train, epochs=EPOCHS,
              batch_size=BATCH_SIZE, validation_split=0.1)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy on MNIST test set: {acc:.4f}")
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

# ----------------------
# 3) Helper: 將學號影像預處理成類似 MNIST 的陣列
# ----------------------
def preprocess_student_image(img_path):
    """
    讀取影像檔案並將其轉換為形狀 (28,28,1)，正歸化為 [0,1]
    接受彩色或灰階 PNG/JPG 格式。可處理白色背景或黑色背景。
    """
    img = Image.open(img_path).convert("L")  # 到灰階
    # 調整大小並保持寬高比：先適應 20x20 尺寸，然後填滿至 28x28 尺寸（MNIST 標準尺寸）
    # 這邊為了簡單起見，直接調整為 28x28 尺寸：
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

    # 轉換為 numpy 數組
    arr = np.array(img).astype("float32")

    # 正歸化到 [0,1]
    arr = arr / 255.0

    # Heuristic：MNIST 數字在黑色（低）背景上顯示為白色（高）。
    # 如果影像是白色背景上的黑色數字，需反轉。
    # 檢查平均亮度：如果平均值 > 0.5，則假設背景為白色 -> 反轉
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    
    # 某些掃描結果可能存在數位稀疏的情況；可選：應用簡單的閾值/對比（非強制）
    # arr = np.where(arr > 0.15, arr, 0.0)

    # 將 dims 擴展為 (28,28,1)
    arr = np.expand_dims(arr, -1)
    return arr

# ----------------------
# 4) 載入學生圖像並預測
# ----------------------
def predict_student_digits(student_dir=STUDENT_DIR):
    if not os.path.isdir(student_dir):
        print(f"Student image folder '{student_dir}' not found.")
        print("請建立資料夾並把每個數字影像放在裡面（支援 png/jpg），檔名建議以 0,1,2,... 或 01 02 03 排序。")
        return

    # 收集圖像文件
    files = [f for f in os.listdir(student_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    if not files:
        print(f"資料夾 {student_dir} 中沒有可辨識的影像檔案。")
        return

    # 將檔案名稱排序，使其順序與學生 ID 順序相對應
    files.sort(key=lambda s: (len(s), s))
    inputs = []
    ordered_files = []
    for fn in files:
        path = os.path.join(student_dir, fn)
        try:
            arr = preprocess_student_image(path)
            inputs.append(arr)
            ordered_files.append(fn)
        except Exception as e:
            print(f"跳過 {fn}（讀取或處理失敗）：{e}")

    if not inputs:
        print("沒有成功讀取任何影像。")
        return

    X = np.stack(inputs, axis=0)  # shape (n,28,28,1)
    preds = model.predict(X)
    pred_labels = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    # Output results: 每個檔名對應的預測數字與信心度
    print("----- 辨識結果 -----")
    for fn, lab, conf in zip(ordered_files, pred_labels, confidences):
        print(f"{fn:20s} -> {lab}    (confidence {conf:.3f})")

    # 組成學號（檔名排序即學號每位）：
    student_id_pred = "".join(str(d) for d in pred_labels)
    print("\n預測的學號（依檔名排序）:", student_id_pred)
    return ordered_files, pred_labels, confidences

if __name__ == "__main__":
    predict_student_digits()

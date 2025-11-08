# 參考 : https://ithelp.ithome.com.tw/articles/10191404

# 導入函式庫
import numpy as np
import os
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import to_categorical  # 用來後續將 label 標籤轉為 one-hot-encoding
from matplotlib import pyplot as plt

# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 建立簡單的線性執行的模型
model = Sequential()
# Add Input layer, 隱藏層(hidden layer) 有 256個輸出變數
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
# Add output layer
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 將 training 的 label 進行 one-hot encoding
y_TrainOneHot = to_categorical(y_train)
y_TestOneHot = to_categorical(y_test)

# 將 training 的 input 資料轉為2維
X_train_2D = X_train.reshape(60000, 28*28).astype('float32')
X_test_2D = X_test.reshape(10000, 28*28).astype('float32')

x_Train_norm = X_train_2D / 255
x_Test_norm = X_test_2D / 255

# 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)

# 顯示訓練成果(分數)
scores = model.evaluate(x_Test_norm, y_TestOneHot)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))

# 預測(prediction)
X = x_Test_norm[0:10, :]
predictions = np.argmax(model.predict(X), axis=-1)
print(predictions)

# 儲存模型設定與權重到目前程式所在資料夾
save_dir = os.path.dirname(os.path.abspath(__file__))  # 目前 py 檔所在資料夾

# 模型結構存檔
json_path = os.path.join(save_dir, "model.config")
with open(json_path, "w") as text_file:
    text_file.write(model.to_json())

# 模型訓練結果存檔 (.weights.h5)
weights_path = os.path.join(save_dir, "model.weights.h5")
model.save_weights(weights_path)

print(f"[Info] Model config saved to: {json_path}")
print(f"[Info] Model weights saved to: {weights_path}")

# 顯示 第一筆訓練資料的圖形，確認是否正確
plt.imshow(X_test[0])
plt.show()

# 查看優化過程
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Train History')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()

# 攝氏與華式溫度轉換
# 載入套件
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 載入上述表格資料
c = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
f = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

# 建立模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Input((1,)),
  tf.keras.layers.Dense(1)
])

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
  loss='mse',
  metrics=['accuracy'])

# 訓練模型
history = model.fit(c, f, epochs=500)

# 損失趨勢繪圖
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.plot(history.history['loss'])
plt.show()

# 模型評分
x_test = np.array([50., 100.],  dtype=float)
y_test = np.array([50., 100.],  dtype=float) * 1.8 + 32
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print(f'loss={loss:4f}, accuracy={accuracy:4f}')

# 顯示模型權重(w, b)
print(f'模型權重(w, b):{model.weights[0][0][0].numpy()}, \
    {model.weights[1][0].numpy()}')

# 繪圖
plt.plot(c, f, 'g')
plt.plot(c, model.predict(c, verbose=False), 'r')
plt.show()


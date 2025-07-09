# 手寫阿拉伯數字辨識
# 載入套件
import tensorflow as tf
import numpy as np

# 載入手寫阿拉伯數字訓練資料(MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# 特徵縮放
x_train, x_test = x_train / 255.0, x_test / 255.0

# one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test_org = y_test.copy()
y_test = tf.keras.utils.to_categorical(y_test)

# 建立模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Input((28, 28)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
model.compile(optimizer='adam',
  loss=tf.keras.losses.MeanSquaredError(), # MSE
  metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, epochs=5)
# 模型評分
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print(f'loss={loss}, accuracy={accuracy}')

pred = model.predict(x_test[:10], verbose=False)
print(f'actual={y_test_org[:10]}\npred  ={np.argmax(pred, axis=-1)}')

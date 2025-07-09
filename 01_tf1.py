# 手寫阿拉伯數字辨識
# 載入套件
import tensorflow as tf
import numpy as np

# 載入手寫阿拉伯數字訓練資料(MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# 特徵縮放
x_train, x_test = x_train / 255.0, x_test / 255.0

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
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, epochs=5)
# 模型評分
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print(f'loss={loss}, accuracy={accuracy}')

pred = model.predict(x_test[:10], verbose=False)
for i in range(10):
    print(f'pred={pred[i]}, max={np.argmax(pred[i])}')
    
model.save('mnist_model.keras')
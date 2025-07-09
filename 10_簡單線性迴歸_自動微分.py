# 載入套件
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.datasets import make_regression
import numpy as np
import tensorflow as tf 

# 生成亂數資料
X1, y= make_regression(n_samples=100, n_features=1, noise=15, bias=50)
X=X1.ravel() # 轉為一維陣列
# print(X, y)  

# 設定圖形更新的頻率
PAUSE_INTERVAL=0.5

# 設定圖形大小
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
          
# 繪製亂數資料散佈圖
plt.scatter(X,y)
line, = ax.plot(X, [0] * len(X), 'r')

# 求預測值(Y hat)
def predict(X):
    return w * X + b  

# 計算損失函數 MSE = ∑(y-ŷ)**2/n
def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

# 定義繪製迴歸線的函數
def draw(w, b):
    # 更新圖形Y軸資料
    y_new = [b + w * xplot for xplot in X]
    line.set_data(X, y_new)  # update the data.
    #ax.cla()
    plt.pause(PAUSE_INTERVAL)

# 定義訓練函數
def train(X, y, epochs=40, lr=0.1):
    current_loss=0                                # 損失函數值
    for epoch in range(epochs):                   # 執行訓練週期
        with tf.GradientTape() as t:              # 自動微分
            t.watch(tf.constant(X))               # 宣告 TensorFlow 常數參與自動微分
            current_loss = loss(y, predict(X))    # 計算損失函數值
        
        dw, db = t.gradient(current_loss, [w, b]) # 取得 w, b 個別的梯度

        # 更新權重：新權重 = 原權重 — 學習率(learning_rate) * 梯度(gradient)
        w.assign_sub(lr * dw) # w -= lr * dw
        b.assign_sub(lr * db) # b -= lr * db
        
        # 更新圖形
        draw(w, b)
        
        # 顯示每一訓練週期的損失函數
        print(f'Epoch {epoch}: Loss: {current_loss.numpy()}') 

# 模型訓練        
# w、b 初始值均設為 0
w = tf.Variable(0.0)
b = tf.Variable(0.0)

# 執行訓練
train(X, y)

# w、b 的最佳解
print(f'w={w.numpy()}, b={b.numpy()}')

# 顯示動畫       
# plt.show()

# 驗證 w、b是否正確：使用Scikit-learn套件比對
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X1, y)
print(f'w={model.coef_[0]}, b={model.intercept_}')

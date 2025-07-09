# 載入套件
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 損失函數
def func(x):
    # return x**2 + 10.0 * tf.sin(x)
    return tf.sin(x)*tf.exp(-0.1*(x-0.6)**2) 

# 模型訓練函數
def train(func, restarts=10, steps=1000, lr=0.01, x_range=(-10.0, 10.0)):
    best_loss = float('inf')
    best_x = None

    # 在[-10.0, 10.0]之間，隨機設定權重初始值
    for i in range(restarts):
        x = tf.Variable(tf.random.uniform(shape=(), minval=x_range[0], maxval=x_range[1]
            , dtype=tf.float64))

        for step in range(steps):
            with tf.GradientTape() as tape:
                loss = func(x)
            grad = tape.gradient(loss, x)
            x.assign_sub(lr * grad)  # Gradient descent update

        final_loss = func(x).numpy()
        print(f"Restart {i+1}: x = {x.numpy():.4f}, loss = {final_loss:.4f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_x = x.numpy()

    return best_x, best_loss

# 模型訓練：呼叫梯度下降法 
best_x, best_loss = train(func)
print(f"\nBest solution: x = {best_x:.4f}, loss = {best_loss:.4f}")

# 繪圖
color = 'r'    
t = np.arange(-5, 5, 0.01)
plt.plot(t, func(t), c='b')
# plt.plot(t, func(best_x[0]), c=color, label='lr={}'.format(lr))    
plt.scatter(best_x, func(best_x), c=color, )    
plt.show()
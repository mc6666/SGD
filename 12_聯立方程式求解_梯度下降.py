# x+y=16
# 10x+25y=250

# 載入套件
import tensorflow as tf
import numpy as np
import random

# 初始化x, y為任意值
x = tf.Variable(random.random())
y = tf.Variable(random.random())

# 梯度下降
EPOCHS = 10000
optimizer = tf.keras.optimizers.Adam(0.1)
previous_loss = np.inf
for i in range(EPOCHS):
    with tf.GradientTape() as tape:
        # 第1題聯立方程式
        y1 = x+y-16
        y2 = 10*x+25*y-250
        
        # 第2題
        # y1 = 2*x + 3*y - 9
        # y2 = x - y - 2
        
        # 第3題
        # y1 = 2*x + 2 - (x-y)
        # y2 = 3*x + 2*y
        
        loss = y1*y1 + y2*y2
        
    # Calculate gradients
    dx, dy = tape.gradient(loss, [x, y])
    # Update model weights
    optimizer.apply_gradients([(dx, x), (dy, y)])
    
    # 提早結束
    if abs(loss - previous_loss) < 10 ** -8:
        print(f'epochs={i+1} 提早結束!!')
        break
    else:
        previous_loss = loss

print(f'x={x.numpy()}')
print(f'y={y.numpy()}')

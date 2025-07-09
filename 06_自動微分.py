# 自動微分(Automatic Differentiation)
# 載入套件
import numpy as np 
import tensorflow as tf 

# 自動微分
# 在 tf.GradientTape 中定義損失函數，使用 g.gradient 取得梯度，使用numpy()轉為NumPy格式
x = tf.Variable(3.0)         # 宣告 TensorFlow 變數(Variable)
with tf.GradientTape() as g: # 自動微分
    y = x * x                # 損失函數
    # y = x * tf.sin(x) ** 2
    
dy_dx = g.gradient(y, x)     # 取得梯度， f'(x) = 2x, x=3 ==> 6
print(dy_dx.numpy())         # 轉換為 NumPy array 格式

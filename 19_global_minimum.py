import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

# 損失函數
def func(x): return np.sin(x)*np.exp(-0.1*(x-0.6)**2)

def train(w_start, epochs, lr):    
    w_list = np.array([])    
    w = tf.Variable(w_start)         # 宣告 TensorFlow 變數(Variable)
    w_list = np.append(w_list, w_start)  
    # 執行N個訓練週期
    for i in range(epochs):         
        with tf.GradientTape() as g: # 自動微分
            y = tf.sin(w)*tf.exp(-0.1*(w-0.6)**2) # 損失函數

        dw = g.gradient(y, w) # 取得梯度
        # 更新權重：新權重 = 原權重 — 學習率(learning_rate) * 梯度(gradient)
        w.assign_sub(lr * dw)  # w -=  dw * lr         
        w_list = np.append(w_list, w.numpy())    
    return w_list

# 權重初始值
# w_start = 0.3  # 找到全局最小值(Global minimum)   
w_start = 5.0  # 找到區域最小值(Local minimum)
# 執行週期數
epochs = 1000 
# 學習率   
lr = 0.01   
# 梯度下降法 
# *** Function 可以直接當參數傳遞 ***
w = train(w_start, epochs, lr=lr) 
print (w)
# 輸出：[-5.     -2.     -0.8    -0.32   -0.128  -0.0512]

color = 'r'    
#plt.plot(line_x, line_y, c='b')    
from numpy import arange
t = arange(-5, 5, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(w, func(w), c=color, label='lr={}'.format(lr))    
plt.scatter(w, func(w), c=color, )    
plt.legend()
plt.show()
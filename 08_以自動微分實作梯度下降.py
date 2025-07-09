# 載入套件
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import tensorflow as tf 

# 修正中文亂碼
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False


# 損失函數為 y=x^2
def func(x): return x ** 2

# @tf.function
def train(w_start, epochs, lr):    
    w_list = np.array([])    
    w = tf.Variable(w_start)         # 宣告 TensorFlow 變數(Variable)
    w_list = np.append(w_list, w_start)  
    # 執行N個訓練週期
    for i in range(epochs):         
        with tf.GradientTape() as g: # 自動微分
            y = w * w                # 損失函數

        dw = g.gradient(y, w) # 取得梯度
        # 更新權重：新權重 = 原權重 — 學習率(learning_rate) * 梯度(gradient)
        w.assign_sub(lr * dw)  # w -=  dw * lr         
        w_list = np.append(w_list, w.numpy())    
    return w_list

# 模型訓練：呼叫梯度下降法 
w_start = -5.0 # 權重初始值
epochs = 150  # 訓練週期數
lr = 0.1      # 學習率
w_list = train(w_start, epochs, lr=lr) 
print (f'w:{np.around(w_list, 2)}')

# 繪圖觀察權重更新過程
color = 'r'    
t = np.arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(w_list, func(w_list), c=color, label='lr={}'.format(lr))    
plt.scatter(w_list, func(w_list), c=color) 
# 繪圖箭頭，顯示權重更新方向
plt.quiver(w_list[0]-0.2, func(w_list[0]), w_list[4]-w_list[0], 
    func(w_list[4])-func(w_list[0]), color='g', scale_units='xy', scale=5)

# 繪圖標題設定
font = {'family': 'Microsoft JhengHei', 'weight': 'normal', 'size': 20}
plt.title('梯度下降法', fontproperties=font)
plt.xlabel('w', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.show()

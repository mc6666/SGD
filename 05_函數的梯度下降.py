# 載入套件
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 修正中文亂碼
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False


# 損失函數為 y=x^2
def func(x): return x ** 2

# 損失函數的一階導數:dy/dx=2*x
def dfunc(x): return 2 * x

def train(w_start, epochs, lr):    
    """ 梯度下降法
        :param w_start: x的起始點    
        :param epochs: 訓練週期    
        :param lr: 學習率    
        :return: x在每次反覆運算後的位置（包括起始點），長度為epochs+1    
     """    
    w_list = np.zeros(epochs+1)    
    w = w_start    
    w_list[0] = w  
    # 執行N個訓練週期
    for i in range(epochs):         
        # 權重的更新W_new
        # W_new = W — learning_rate * gradient        
        w -=  dfunc(w) * lr         
        w_list[i+1] = w   
    return w_list

# 模型訓練：呼叫梯度下降法 
w_start = -5 # 權重初始值
epochs = 150 # 訓練週期數
lr = 0.1     # 學習率
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

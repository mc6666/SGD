# PyTorch自動微分的語法
import torch       # 載入套件

x = torch.tensor(3.0, requires_grad=True)  # 設定 x 參與自動微分
y=x*x                 # 損失函數
y.backward()          # 反向傳導
print(x.grad.numpy()) # 取得梯度

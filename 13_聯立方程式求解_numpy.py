# x+y=16
# 10x+25y=250

# 載入套件
import numpy as np

# 定義方程式的 A、B
A = np.array([[1 , 1], [10, 25]])
B = np.array([16, 250])
print(B.reshape(2, 1))

# np.linalg.solve：線性代數求解
print('\n線性代數求解：')
print(np.linalg.solve(A, B))

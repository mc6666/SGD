# 載入套件
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成訓練資料
np.random.seed(0)
x_data = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
y_data = np.sin(x_data) + 0.1 * np.random.randn(*x_data.shape)

# 模型
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='tanh', input_shape=(1,)),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])
    return model

# 模型訓練函數
def train(x, y, restarts=5, epochs=200, batch_size=32):
    best_loss = float('inf')
    best_model = None

    for i in range(restarts):
        print(f"Restart {i + 1}/{restarts}")
        model = build_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss='mse')

        history = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
        final_loss = history.history['loss'][-1]

        print(f"  Final loss: {final_loss:.4f}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model

    return best_model, best_loss

# 特徵縮放
x_norm = (x_data - np.mean(x_data)) / np.std(x_data)

# 模型訓練
best_model, best_loss = train(x_norm, y_data, restarts=5)
print(f"\nBest loss after restarts: {best_loss:.4f}")

y_pred = best_model.predict(x_norm)

# 繪圖
plt.figure(figsize=(10, 5))
plt.scatter(x_data, y_data, label='Noisy Data', s=10)
plt.plot(x_data, y_pred, color='red', label='NN Prediction', linewidth=2)
plt.legend()
plt.title("Fitting a Noisy Sine Wave using Neural Network")
plt.grid(True)
plt.show()

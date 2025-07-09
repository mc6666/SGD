# 改變與取得學習率

# 載入套件
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定義函數在訓練過程中顯示學習率
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._get_current_learning_rate()
    return lr

# 模型定義
model = tf.keras.models.Sequential([tf.keras.layers.Input((20,)), tf.keras.layers.Dense(10)])
optimizer = tf.keras.optimizers.Adam()
lr_metric = get_lr_metric(optimizer)
model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy', lr_metric])

# 前 10 個 epochs 的學習率為固定值，之後逐次衰減
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return np.float64(lr * tf.math.exp(-0.1)) # 學習率每週期約減少1/10

# 在 callback 中改變學習率
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# 模型訓練
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=15, callbacks=[callback], verbose=2)

# 對訓練過程的學習率繪圖
plt.figure(figsize=(8, 6))
plt.plot(history.history['lr'], 'r')
plt.show()
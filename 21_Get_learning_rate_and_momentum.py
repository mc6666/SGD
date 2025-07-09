# 改變與取得學習率

# 載入套件
# 載入套件
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
    
# 定義函數在訓練過程中顯示學習率
class VelocityLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.var_list = self.model.trainable_weights

    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        # if epoch == 0:
            # print(f"var names: {optimizer.variables}")
        print(f"\nEpoch {epoch+1} velocities (mean absolute):")
        for var in optimizer.variables:
            if 'momentum' in var.name:
                mean_vel = tf.reduce_mean(tf.abs(var)).numpy()
                print(f"  {var.name}: {mean_vel:.6e}")


# 模型定義
model = tf.keras.models.Sequential([tf.keras.layers.Input((20,)), tf.keras.layers.Dense(10)])
# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# 在 callback 中改變學習率
# scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    # initial_learning_rate=0.01,
    # decay_steps=100,
    # decay_rate=0.9,
    # staircase=True
# )
# callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# 模型訓練
callback = VelocityLogger()
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=15, callbacks=[callback], verbose=2)

# 對訓練過程的學習率繪圖
# plt.figure(figsize=(8, 6))
# plt.plot(history.history['lr'], 'r')
# plt.show()
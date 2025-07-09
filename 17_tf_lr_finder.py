# 尋找最佳學習率初始值

# 載入套件
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from lr_finder import LRFinder

# 載入訓練資料
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()
x_train, x_valid = x_train / 255.0, x_valid / 255.0
x_train = x_train[..., tf.newaxis]
x_valid = x_valid[..., tf.newaxis]
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32)

# 模型定義
def build_model():
    return tf.keras.models.Sequential([
        Conv2D(32, 3, activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(10, activation='softmax')
    ])

# 模型訓練
lr_finder = LRFinder()
model = build_model()
adam = tf.keras.optimizers.Adam(learning_rate=1e-1)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=5, callbacks=[lr_finder])

# 對訓練過程的學習率繪圖
lr_finder.plot()
plt.axvline(1e-3, c='r');
plt.show()

# 模型評分
_, accuracy = model.evaluate(valid_ds, verbose=False)
print(f'accuracy={accuracy}')


# 採用最佳學習率重新訓練
model = build_model() # reinitialize model
adam = tf.optimizers.Adam(1e-3)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
_ = model.fit(train_ds, validation_data=valid_ds, epochs=5, verbose=True)
_, accuracy = model.evaluate(valid_ds, verbose=False)
print(f'最佳學習率 accuracy={accuracy}')

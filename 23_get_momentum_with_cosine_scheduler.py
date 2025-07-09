# 手寫阿拉伯數字辨識
# 載入套件
import tensorflow as tf
# 載入手寫阿拉伯數字訓練資料(MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# 特徵縮放
x_train, x_test = x_train / 255.0, x_test / 255.0

# 建立模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Input((28, 28)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 設定學習率設定為變動值
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.001,
    first_decay_steps=10,
    t_mul=2.0,
    m_mul=0.9,
    alpha=0.0
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.CategoricalCrossentropy() #from_logits=True)

# 訓練模型
epochs = 10
batch_size = 1000
for epoch in range(epochs):
    for batch in range(x_train.shape[0] // batch_size):
        x_batch = x_train[batch * batch_size : (batch + 1) * batch_size]
        y_batch = y_train[batch * batch_size : (batch + 1) * batch_size]

        with tf.GradientTape() as tape:
            # Forward pass
            pred = model(x_batch)
            # Calculate the loss
            loss = loss_fn(y_batch, pred)

        # Calculate gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Update model weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(f"Epoch {epoch+1}, Loss: {loss.numpy()}, Learning rate: {optimizer._get_current_learning_rate().numpy()}")
    for var in optimizer.variables:
        if 'momentum' in var.name:
            mean_vel = tf.reduce_mean(tf.abs(var)).numpy()
            print(f"  {var.name}: {mean_vel:.6e}")

# 模型評分
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print(f'loss={loss}, accuracy={accuracy}')    
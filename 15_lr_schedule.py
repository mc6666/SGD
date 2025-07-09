import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Create a learning rate schedule
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=initial_learning_rate,
    first_decay_steps=1000,   # Number of steps before first restart
    t_mul=2.0,                 # Multiply decay steps by this value after each restart
    m_mul=0.9,                 # Multiplier for initial_learning_rate after each restart
    alpha=0.0                  # Minimum learning rate value as a fraction of initial_learning_rate
)

# 2. Create optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 3. Compile your model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 4. Train the model (dummy data used here for illustration)
import numpy as np
x_train = np.random.rand(5000, 32)
y_train = np.random.rand(5000, 1)

history = model.fit(x_train, y_train, epochs=10, batch_size=32)

# 5. Plot the learning rate schedule for visualization
steps = tf.range(10000, dtype=tf.float32)
lrs = [lr_schedule(step).numpy() for step in steps]

plt.figure(figsize=(10, 4))
plt.plot(steps, lrs)
plt.title("Cosine Decay with Restarts Learning Rate Schedule")
plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.show()

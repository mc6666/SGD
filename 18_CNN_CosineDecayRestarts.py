import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the cifar10  dataset
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0
# x_train = x_train[..., tf.newaxis].astype("float32")
# x_val = x_val[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).repeat().batch(1000)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).repeat().batch(1000)

# Build a simple CNN model
class SimpleCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.out = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)

model = SimpleCNN()

# Define cosine decay with restarts
initial_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=initial_lr,
    first_decay_steps=1000,
    t_mul=2.0,
    m_mul=0.9,
    alpha=0.0
)

# Optimizer with scheduled learning rate
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = tf.keras.optimizers.Adam()

# Loss and metrics
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Training loop
epochs = 5
global_step = 0

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training
    for step, (x_batch, y_batch) in enumerate(train_ds):
        if step >= 100: break
        global_step += 1
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        train_acc_metric.update_state(y_batch, logits)
        
        if step % 10 == 0:
            current_lr = optimizer._get_current_learning_rate().numpy() #_decayed_lr(tf.float32).numpy()
            print(f"Step {step}, Loss: {loss:.4f}, LR: {current_lr:.6f}")
    
    train_acc = train_acc_metric.result()
    print(f"Training accuracy over epoch: {train_acc:.4f}")
    train_acc_metric.reset_state()

    # Validation
    for step, (x_batch_val, y_batch_val) in enumerate(val_ds):
        if step >= 10: break
        val_logits = model(x_batch_val, training=False)
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f"Validation accuracy: {val_acc:.4f}")

# Plot learning rate schedule
steps = tf.range(3000, dtype=tf.float32)
lrs = [lr_schedule(step).numpy() for step in steps]

plt.plot(steps, lrs)
plt.title("Cosine Decay with Restarts Learning Rate")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.show()

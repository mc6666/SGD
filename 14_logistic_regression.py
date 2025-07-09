# Soucre：https://stackoverflow.com/questions/56907971/logistic-regression-using-tensorflow-2-0
# 載入套件
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import tensorflow as tf

# 載入訓練資料(MNIST)及特徵縮放
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255., x_test/255.

# 切割訓練資料及驗證資料，並將訓練資料及測試資料轉為一維
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)
x_train = tf.reshape(x_train, shape=(-1, 784))
x_test  = tf.reshape(x_test, shape=(-1, 784))

# 初始化權重及偏差為任意值
weights = tf.Variable(tf.random.normal(shape=(784, 10), dtype=tf.float64))
biases  = tf.Variable(tf.random.normal(shape=(10,), dtype=tf.float64))

# 定義 y = wx + b
def logistic_regression(x):
    lr = tf.add(tf.matmul(x, weights), biases)
    #return tf.nn.sigmoid(lr)
    return lr


# 定義交叉熵(Cross entropy)
def cross_entropy(y_true, y_pred):
    y_true = tf.one_hot(y_true, 10) # one-hot encoding
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(loss)

# 定義準確率
def accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.int32)
    preds = tf.cast(tf.argmax(y_pred, axis=1), dtype=tf.int32)
    preds = tf.equal(y_true, preds)
    return tf.reduce_mean(tf.cast(preds, dtype=tf.float32))

# 梯度下降
def grad(x, y):
    with tf.GradientTape() as tape:
        y_pred = logistic_regression(x)
        loss_val = cross_entropy(y, y_pred)
    return tape.gradient(loss_val, [weights, biases])

# 訓練模型
n_batches = 10000
learning_rate = 0.01
batch_size = 128

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat().shuffle(10000).batch(batch_size).prefetch(1)

optimizer = tf.optimizers.SGD(learning_rate)

for batch_no, (batch_xs, batch_ys) in enumerate(dataset.as_numpy_iterator()):
    if batch_no >= n_batches: break
    gradients = grad(batch_xs, batch_ys)
    optimizer.apply_gradients(zip(gradients, [weights, biases]))

    y_pred = logistic_regression(batch_xs)
    loss = cross_entropy(batch_ys, y_pred)
    acc = accuracy(batch_ys, y_pred)
    if (batch_no+1) % 100 == 0: # 每100批顯示訓練結果
        print(f"Batch number: {batch_no+1}, loss: {loss}, accuracy: {acc}")
        
# 模型評分
y_pred = logistic_regression(x_test)
loss = cross_entropy(y_test, y_pred)
acc = accuracy(y_test, y_pred)
print(f"accuracy: {acc}")
        
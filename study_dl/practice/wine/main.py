import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name = 'target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

n_input = X_train.shape[1]
n_hidden = 26
n_output = 3

X_train = tf.constant(X_train, dtype = tf.float32)
X_test = tf.constant(X_test, dtype = tf.float32)
y_train = tf.cast(tf.one_hot(y_train, depth=n_output), tf.float32)
y_test  = tf.cast(tf.one_hot(y_test,  depth=n_output), tf.float32)

w1 = tf.Variable(tf.random.normal([n_input, n_hidden]), stddev = 0.1)
b1 = tf.Variable(tf.zeros(n_hidden))

w2 = tf.Variable(tf.random.normal([n_hidden, n_output]), stddev = 0.1)
b2 = tf.Variable(tf.zeros(n_output))

epochs = 200
lr = 0.01
epsilon = 1e-9

def relu(x):
    return tf.maximum(x, 0.0)

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        z1 = tf.matmul(X_train, w1)+b1
        a1 = relu(z1)

        z2 = tf.matmul(a1,w2)+b2
        y_pred = tf.nn.softmax(z2)
        
        loss = -tf.reduce_mean(tf.reduce_sum(y_train * tf.math.log(y_pred + epsilon), axis=1))

    grads = tape.gradient(loss, [w1,b1,w2,b2])

    w1.assign_sub(lr * grads[0])
    b1.assign_sub(lr * grads[1])
    w2.assign_sub(lr * grads[2])
    b2.assign_sub(lr * grads[3])
    
    y_pred_class = tf.argmax(y_pred, axis=1)
    y_true_class = tf.argmax(y_train, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_class, y_true_class), tf.float32))

    if epoch%20 == 0:
        print(f"Эпоха {epoch}: loss={loss.numpy():.4f}, acc={accuracy.numpy():.4f}")

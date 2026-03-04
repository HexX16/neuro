import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Синтетические данные: 200 точек, 2 класса
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
X = tf.constant(X, dtype=tf.float32)
y = tf.constant(y.reshape(-1, 1), dtype=tf.float32)

plt.scatter(X[:,0], X[:,1], c=y.numpy().ravel(), cmap='bwr')
plt.show()

n_input = 2
n_hidden1 = 10
n_hidden2 = 5
n_output = 1

w1 = tf.Variable(tf.random.normal([n_input, n_hidden1], stddev=0.1))
b1 = tf.Variable(tf.zeros(n_hidden1))

w2 = tf.Variable(tf.random.normal([n_hidden1, n_hidden2], stddev=0.1))
b2 = tf.Variable(tf.zeros(n_hidden2))

w3 = tf.Variable(tf.random.normal([n_hidden2, n_output], stddev=0.1))
b3 = tf.Variable(tf.zeros(n_output))

lr = 0.15
epsilon = 1e-9
epochs = 500

def relu(x):
    return tf.maximum(x, 0.0)

def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        z1 = tf.matmul(X, w1) + b1
        a1 = relu(z1)

        z2 = tf.matmul(a1, w2) + b2
        a2 = relu(z2)

        z3 = tf.matmul(a2, w3) + b3
        y_pred = sigmoid(z3)

        loss = -tf.reduce_mean(y * tf.math.log(y_pred + epsilon) + (1 - y) * tf.math.log(1 - y_pred + epsilon))
    
    grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])

    w1.assign_sub(lr * grads[0])
    b1.assign_sub(lr * grads[1])
    w2.assign_sub(lr * grads[2])
    b2.assign_sub(lr * grads[3])
    w3.assign_sub(lr * grads[4])
    b3.assign_sub(lr * grads[5])

    if epoch%50==0:
        print(f"Эпоха {epoch}: loss={loss.numpy():.4f}")

# Forward pass для визуализации (прямой проход)
z1 = tf.matmul(X, w1) + b1
a1 = relu(z1)

z2 = tf.matmul(a1, w2) + b2
a2 = relu(z2)

z3 = tf.matmul(a2, w3) + b3
y_pred = sigmoid(z3)

y_class = tf.cast(y_pred >= 0.5, tf.float32)

plt.scatter(X[:,0], X[:,1], c=y_class.numpy().ravel(), cmap='bwr')
plt.title('Предсказанные классы')
plt.show()
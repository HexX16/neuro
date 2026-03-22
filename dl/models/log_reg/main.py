import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

e = 10**-9
lr = 0.01

x = tf.constant(tf.random.normal([100, 1], dtype = tf.float32))
y = tf.cast(x > 0, tf.float32)

w = tf.Variable(tf.random.normal([1,1]))
b = tf.Variable(tf.zeros([1]))

for step in range(200):
    with tf.GradientTape() as tape:
        y_pred = 1 / (1 + tf.exp(-(tf.matmul(x, w)+b))) #или так:  y_pred = tf.sigmoid(tf.matmul(x, w) + b)
        loss = -tf.reduce_mean(y * tf.math.log(y_pred + e) + (1 - y) * tf.math.log(1 - y_pred + e))
    grads = tape.gradient(loss, [w, b])
    w.assign_sub(grads[0] * lr)
    b.assign_sub(grads[1] * lr)
    if step % 20 == 0:
        print(f"Шаг {step:3d}: loss={loss.numpy():.4f}, w={w.numpy()[0][0]:.3f}, b={b.numpy()[0]:.3f}")


y_pred_class = tf.cast(y_pred >= 0.5, tf.float32)

# График
plt.scatter(x, y, label='Данные')
plt.scatter(x, y_pred_class, color='red', label='Предсказания', marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
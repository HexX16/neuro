import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 50).reshape(-1,1)
y = 2 * x + 1 + np.random.randn(*x.shape) * 0.5

x_tf = tf.constant(x, dtype=tf.float32)
y_tf = tf.constant(y, dtype=tf.float32)

w = tf.Variable(tf.random.normal([1, 1]))
b = tf.Variable(tf.zeros([1]))
lr = 0.01

for step in range(200):
    with tf.GradientTape() as tape:
        y_pred = tf.matmul(x_tf, w) + b
        loss = tf.reduce_mean((y_tf - y_pred) ** 2)
    
    grads = tape.gradient(loss, [w,b])

    w.assign_sub(grads[0]*lr)
    b.assign_sub(grads[1]*lr)
    
    if step % 20 == 0:
        print(f"Шаг {step:3d}: loss={loss.numpy():.4f}, w={w.numpy()[0][0]:.3f}, b={b.numpy()[0]:.3f}")

plt.scatter(x, y, label='Данные')
plt.plot(x, x * w.numpy()[0][0] + b.numpy()[0], color='red', label='Модель')
plt.legend()
plt.show()
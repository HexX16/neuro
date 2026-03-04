import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

x = np.random.rand(100)
true_w = 4
true_b = 3
y = x * true_w + true_b + np.random.randn(100)

w = np.random.rand()
b = np.random.rand()

def predict(x, w, b):
    y_pred = x*w + b
    return y_pred

def loss(y, y_pred):
    return np.mean((y_pred - y)**2)

def grad(x, y, y_pred):
    dw = np.mean(2*x*(y_pred-y))
    db = np.mean(2*(y_pred-y))
    return dw, db

def train(x, y, w, b, epochs=1000, lr = 0.01):
    for i in range(epochs):
        y_pred = predict(x, w, b)
        dw, db = grad(x, y, y_pred)
        w -= dw*lr
        b -= db*lr

        if i%100==0:
            print(f"{i}: loss = {loss(y, y_pred)}, w = {w}, b = {b}")

    return w,b

train(x, y, w, b)
print(true_w,true_b)
        


# plt.scatter(x, y)   
# plt.show()
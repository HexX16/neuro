import numpy as np
import pandas as pd

class LinearRegressionOneFeature:
    def __init__(self):
        self.w = np.random.rand()
        self.b = np.random.rand()
        self.learning_rate = 0.01
        self.epochs = 1000
    
    def predict(self, x):
        y_pred = x*self.w + self.b
        return y_pred
    
    def calc_mse(self, y, y_pred):
        mse = np.mean((y_pred-y)**2)
        return mse
    
    def gradient(self, x, y, y_pred):
        dw = 2*np.mean(x*(y_pred-y)) 
        db = 2*np.mean(y_pred-y)
        return dw, db
    
    def fit(self, x, y):

        for i in range(self.epochs):
            y_pred = self.predict(x)
            dw, db = self.gradient(x, y, y_pred)
            self.w -= dw*self.learning_rate
            self.b -= db*self.learning_rate
            if i%100==0:
                print(f"{i}: loss = {self.calc_mse(y, y_pred)}, w = {self.w}, b = {self.b}")

class LinearRegression:
    def __init__(self):
        self.learning_rate = 0.01
        self.epochs = 1000
        self.mean = None
        self.std = None
    
    def predict(self, x):
        x_norm = self.normalize(x)
        y_pred = x_norm.dot(self.w) + self.b
        return y_pred
    
    def normalize(self, x, fit = False):
        if fit:
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
        return (x - self.mean) / (self.std + 1e-8)

    def calc_mse(self, y, y_pred):
        mse = np.mean((y_pred-y)**2)
        return mse
    
    def gradient(self, x, y, y_pred):
        n = len(y)
        dw = (2/n) * x.T.dot(y_pred - y)
        db = 2*np.mean(y_pred-y)
        return dw, db
    
    def fit(self, x, y):
        self.w = np.random.rand(x.shape[1])
        self.b = np.random.rand()
        x_norm = self.normalize(x, fit=True)
        for i in range(self.epochs):
            y_pred = x_norm.dot(self.w) + self.b
            dw, db = self.gradient(x_norm, y, y_pred)
            self.w -= dw*self.learning_rate
            self.b -= db*self.learning_rate
            if i%100==0:
                print(f"{i}: MSE = {self.calc_mse(y, y_pred)}, w = {self.w}, b = {self.b}")



        
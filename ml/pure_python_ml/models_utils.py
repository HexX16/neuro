import numpy as np
from statistics import mode  


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

class LogisticRegression:
    def __init__(self):
        self.learning_rate = 0.01
        self.epochs = 1000
        self.mean = None
        self.std = None

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def normalize(self, x, fit = False):
        if fit:
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
        return (x - self.mean) / (self.std + 1e-8)

    def predict(self, x):
        x_norm = self.normalize(x)
        y_pred = self.sigmoid(x_norm.dot(self.w)+self.b)
        return (y_pred > 0.5).astype(int)
    
    def calc_cross_entropy(self, y, y_pred):
        loss = -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
        return loss

    def gradient(self, x, y, y_pred):
        m = x.shape[0]
        dw = x.T.dot(y_pred-y)/m
        db = np.sum(y_pred-y)/m
        return dw,db
    
    def fit(self, x, y):
        self.w = np.random.rand(x.shape[1])
        self.b = np.random.rand()
        x_norm = self.normalize(x, fit=True)
        for i in range(self.epochs):
            y_pred = self.sigmoid(x_norm.dot(self.w)+self.b)
            dw, db = self.gradient(x_norm, y, y_pred)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if i%100==0:
                print(f"{i}: cross_entropy = {self.calc_cross_entropy(y, y_pred)}, w = {self.w}, b = {self.b}")

class KNN():
    def __init__(self):
        self.k = 3

    def euclid_dist(self, X_train, X_test):
        dist = np.sqrt(np.sum((X_train-X_test)**2, axis=1))
        return dist
    
    def predict(self, X_train, X_test, y_train):
        dist = self.euclid_dist(X_train, X_test)
        indices = np.argsort(dist)[:self.k]
        neighbors = y_train[indices]
        return mode(neighbors)
    
class DesicionTree():
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts/len(y)
        return 1 - np.sum(p**2)
    
    def best_split(self, X, y):
        best_gini = 1
        best_idx, best_thr = None, None
        for idx in range(X.shape[1]):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left_mask = X[:, idx] <= thr
                right_mask = X[:, idx] > thr
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                gini_left = self.gini(y[left_mask])
                gini_right = self.gini(y[right_mask])
                gini_split = (len(y[left_mask])/len(y)) * gini_left + (len(y[right_mask])/len(y)) * gini_right
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_idx = idx
                    best_thr = thr
        return best_idx, best_thr
    
    def build_tree(self, X, y, depth=0):
    # 1. Проверяем, все ли объекты одного класса
        if len(np.unique(y)) == 1:
            return Node(value=y[0])  # создаём лист

        # 2. Проверяем ограничение по глубине
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=np.bincount(y).argmax())  # лист с самым частым классом

        # 3. Находим лучший сплит
        idx, thr = self.best_split(X, y)
        if idx is None:  # если делить нельзя
            return Node(value=np.bincount(y).argmax())

        # 4. Делим данные на левую и правую ветки
        left_mask = X[:, idx] <= thr
        right_mask = X[:, idx] > thr

        # 5. Рекурсивно строим левое и правое поддеревья
        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        # 6. Возвращаем узел с поддеревьями
        return Node(feature_idx=idx, threshold=thr, left=left, right=right)

    def fit(self, X, y, max_depth=None):
        self.root = self.build_tree(X, y, depth=0)

    def predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

    def predict(self, X):
        return np.array([self.predict_single(x, self.root) for x in X])
    
class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # индекс признака, по которому делим
        self.threshold = threshold      # порог, по которому делим
        self.left = left                # левое поддерево
        self.right = right              # правое поддерево
        self.value = value              # если это лист, здесь класс
    

    
    


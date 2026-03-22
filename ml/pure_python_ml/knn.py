''' по росту и весу определить тип человека
0 - лёгкий
1 - тяжёлый
'''

import numpy as np
import matplotlib.pyplot as plt
from models_utils import KNN

X_train = [
    [150, 50],
    [160, 55],
    [170, 65],
    [180, 80],
    [175, 75],
    [165, 60],
    [155, 52],
    [185, 90]
]
X_train = np.array(X_train)

y_train = [0,0,0,1,1,0,0,1]
y_train = np.array(y_train)

X_test = [180, 75]

X_test = np.array(X_test)

model = KNN()
y_pred = model.predict(X_train, X_test, y_train)

print(y_pred)
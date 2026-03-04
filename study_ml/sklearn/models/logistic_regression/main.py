import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Данные: часы подготовки (X) и факт поступления (y)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0 - не поступил, 1 - поступил

model = LogisticRegression()
model.fit(X,y)

# Прогноз вероятности для студента, который учил 4.5 часа
prob = model.predict_proba([[4.5]])[0, 1]
print("Вероятность поступления при 4.5 часах:", prob)

# Визуализация сигмоиды
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
probs = model.predict_proba(X_test)[:,1]

plt.scatter(X, y, color = 'blue', label = 'Данные')
plt.plot(X_test, probs, color = 'red', label = 'Сигмоида')
plt.xlabel("Часы подготовки")
plt.ylabel("Вероятность поступления")
plt.legend()
plt.show()
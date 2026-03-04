import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Данные: мощность двигателя, макс скорость
X = np.array([100, 150, 200, 250, 300, 350, 400]).reshape(-1, 1)
y = np.array([160, 170, 200, 200, 210, 215, 230])

# Создаём и обучаем модель
model = LinearRegression()
model.fit(X, y)

# Коэффициенты
print("Коэффициент w1:", model.coef_[0])
print("Свободный член w0 (b):", model.intercept_)

# Прогноз для квартиры тестовых данных (X_test)
X_test = np.array([168, 330, 600]).reshape(-1, 1)
predicts = model.predict(X_test)
print(predicts)

plt.scatter(X, y, color = 'blue', label = 'Данные')
plt.scatter(X_test, predicts, color = 'green', label = 'Предсказания')
plt.plot(X, model.predict(X), color = 'red', label = 'Линия регрессии')
plt.plot(X_test, predicts, color = 'green', label = 'Предсказания', linestyle = '--')
plt.xlabel('Мощность двигателя (лс)')
plt.ylabel('Макс скорость')
plt.legend()
plt.show()



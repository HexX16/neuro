'''
Сгенерируй случайную матрицу data размером 100x5, где строки — это наблюдения, а столбцы — признаки.

Для каждого столбца посчитай: среднее, стандартное отклонение, минимум и максимум.

Нормализуй каждый столбец по формуле

Создай случайный вектор весов w размерности 5 и посчитай предсказания y = X @ w.

Найди корреляцию между каждой колонкой исходной матрицы и y.
'''


import numpy as np

data = np.random.rand(100,5)

mean = data.mean(axis=0)
std = data.std(axis=0)
minimum = data.min(axis=0)
maximum = data.max(axis=0)

norm_data = (data-mean)/std

w = np.random.rand(5)

predicts = norm_data @ w

correlations = np.corrcoef(data.T, predicts)[-1, :-1]

print(norm_data)
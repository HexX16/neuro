'''
Создай DataFrame df с 100 строк и колонками:

'Name' — случайное имя из списка ['Alice','Bob','Charlie','Diana']

'Age' — случайное целое от 18 до 60

'Height' — случайное число от 150 до 200

'Weight' — случайное число от 50 до 100

'Score' — случайное число от 0 до 100

Сделай следующие действия:

Найди средний Score и Age для каждого имени (groupby)

Отфильтруй строки, где Score > 70 и Age < 30

Добавь новую колонку 'BMI' = Weight / (Height/100)^2

Найди средний BMI для каждой группы Name
'''

import pandas as pd
import numpy as np

np.random.seed(42)
names = ['Alice','Bob','Charlie','Diana']
random_names = np.random.choice(names, size = 100)
ages = np.random.randint(18,60,100)
heights = np.random.randint(150,200,100)
weights = np.random.randint(50,100,100)
scores = np.random.randint(0,100,100)
df = pd.DataFrame({
    'Name': random_names,
    'Age': ages,
    'Height': heights,
    'Weight': weights,
    'Score': scores
})
mean_score_age = df.groupby('Age')['Score'].mean()
filt_score_age = df[(df['Age']<30) & (df['Score']>70)]
df['BMI'] = df['Weight'] / ((df['Height']/100)**2)
mean_BMI = df.groupby('Name')['BMI'].mean()

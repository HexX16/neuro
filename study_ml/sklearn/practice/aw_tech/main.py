import pandas as pd
from DomainScaler import DomainScaler
from metering import metering_max, metering, metering_min
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from termcolor import colored
import matplotlib.pyplot as plt


metering_df = pd.DataFrame(metering)
cv = KFold(n_splits=3, shuffle=True, random_state=42)


#Загрузка данных таблицы
df = pd.read_excel('files\\tech_augmented_simple.xlsx')
y = df['technique']
X = df.drop(columns='technique', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Предобработка
scaler = DomainScaler(per_feature_min=metering_min, per_feature_max=metering_max, feature_range=(0,5))
metering_scaled = scaler.fit_transform(metering_df)
metering_scaled_df = pd.DataFrame(metering_scaled, columns=metering_df.columns)

pipeline = Pipeline([
    ('model', KNeighborsClassifier(weights='distance', n_neighbors=9)) 
])

params = {
    'model__metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
}

grid = GridSearchCV(pipeline, param_grid=params, cv=cv, scoring='accuracy', n_jobs = -1)
grid.fit(X_train, y_train)
print("Лучшие параметры:", grid.best_params_, '\n\n\n')
print(colored("Лучшая точность на CV:", 'red'), grid.best_score_)

model = grid.best_estimator_
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(colored("\nТочность на тесте:",'red'), accuracy)

y_pred = model.predict(metering_scaled_df)
print(colored("\nПредсказанная техника:", 'red'), y_pred[0], '\n')


print(metering_df)
print(metering_scaled_df)

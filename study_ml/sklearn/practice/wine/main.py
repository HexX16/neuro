from sklearn.datasets import load_wine
import pandas as pd
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from termcolor import colored
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
data = load_wine()
cv = KFold(n_splits=5, shuffle=True, random_state=42)

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
type = df['target']
df = df.drop(columns=['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(df, type, test_size=0.2, random_state=42)

num_cols = list(df.columns)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols)
])
best_models = {}

models = {
    'Logistic Regression': (
        LogisticRegression(max_iter=10000),
        {
            'model__C': [0.01, 0.1, 1, 10, 100],            # коэффициент регуляризации (чем меньше - сильнее регуляризация)
            'model__penalty': ['l2'],                  # тип регуляризации: L1 — Lasso, L2 — Ridge
        }
    ),
    'Random Forest': (
        RandomForestClassifier(),
        {
            'model__n_estimators': [50, 100, 200],           # количество деревьев в лесу
            'model__max_depth': [None, 10, 20, 30],          # максимальная глубина дерева (None — без ограничения)
            'model__min_samples_split': [2, 5, 10],          # минимальное число образцов для разбиения узла
            'model__min_samples_leaf': [1, 2, 4]              # минимальное число образцов в листовом узле
        }
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(),
        {
            'model__n_estimators': [100, 200],                # количество слабых моделей (деревьев)
            'model__learning_rate': [0.01, 0.1, 0.2],         # скорость обучения (чем меньше — тем медленнее обучение)
            'model__max_depth': [3, 5, 7],                    # максимальная глубина каждого дерева
            'model__subsample': [0.8, 1.0]                    # доля выборки для построения каждого дерева (для стохастичности)
        }
    ),
    'Decision Tree': (
        DecisionTreeClassifier(),
        {
            'model__max_depth': [None, 10, 20, 30],           # максимальная глубина дерева
            'model__min_samples_split': [2, 5, 10],           # минимальное число образцов для разбиения
            'model__min_samples_leaf': [1, 2, 4]               # минимальное число образцов в листовом узле
        }
    ),
    'SVM': (
        SVC(probability=True),
        {
            'model__C': [0.1, 1, 10, 100],                     # коэффициент регуляризации (больше — меньше регуляризации)
            'model__kernel': ['linear', 'rbf'],                # ядро: линейное или радиальное базисное (RBF)
            'model__gamma': ['scale', 'auto']                   # параметр ядра для rbf ('scale' — 1/(n_features*X.var))
        }
    ),
    'K-Nearest Neighbors': (
        KNeighborsClassifier(),
        {
            'model__n_neighbors': [3, 5, 7, 9],                 # количество соседей для классификации
            'model__weights': ['uniform', 'distance'],          # вес соседей: одинаковый или обратно пропорционален расстоянию
            'model__metric': ['euclidean', 'manhattan']         # метрика расстояния: Евклидова или Манхэттенская
        }
    )
}

for name, (model, param) in models.items():
    pipeline = Pipeline([
        ('scaler', preprocessor),
        ('model', model)
    ])
    accuracy_model = cross_val_score(pipeline, X_train, y_train, cv=cv, n_jobs=-1, scoring='accuracy')
    mean_accuracy_model = np.mean(accuracy_model)
    print(colored(f"\n{name}:", 'red'))
    print(colored('Точность до подбора гиперпараметров:', 'yellow'), f'{mean_accuracy_model:.3F}')    
    grid = GridSearchCV(pipeline, param, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    accuracy_best_model = cross_val_score(best_model, X_train, y_train, cv=cv, n_jobs=-1, scoring='accuracy')
    mean_accuracy_best_model = np.mean(accuracy_best_model)
    print(colored('Точность после подбора гиперпараметров:', 'yellow'), f'{mean_accuracy_best_model:.3F}')

#Финальные предсказания
print("\n\n\n\nФинальная оценка на test:")
for name, model in best_models.items():
    model.fit(X_train, y_train)
    predicts = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicts)
    cm = confusion_matrix(y_test, predicts)
    print(colored(f"\n{name}:", 'red'))
    print(f"Точность = {accuracy:.3f}")
    print(f"Матрица ошибок:\n{cm}")
    print(classification_report(y_test, predicts))
    
    






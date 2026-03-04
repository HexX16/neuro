from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
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
from sklearn.metrics import classification_report, roc_auc_score


titanic = fetch_openml('titanic', version=1, as_frame=True)
X = titanic.data
y = titanic.target
X = X.drop(columns = ['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'])
cv= KFold(n_splits=5, shuffle=True, random_state=42)

print(titanic)

num_cols = ['age', 'sibsp', 'parch', 'fare']
cat_cols = ['pclass', 'sex', 'embarked']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder())
    ]), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

best_models = {}

models = {
    'Logistic Regression': (
        LogisticRegression(max_iter=20000),
        {'model__C': [0.01, 0.1, 1, 10, 100], 'model__penalty': ['l2']}
    ),
    'Random Forest': (
        RandomForestClassifier(),
        {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        }
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(),
        {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5],
            'model__subsample': [0.8, 1.0]
        }
    ),
    'Decision Tree': (
        DecisionTreeClassifier(),
        {
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        }
    ),
    'SVM': (
        SVC(probability=True),
        {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__kernel': ['linear', 'rbf'],
            'model__gamma': ['scale', 'auto']
        }
    ),
    'K-Nearest Neighbors': (
        KNeighborsClassifier(),
        {
            'model__n_neighbors': [2, 3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan']
        }
    )
}


for name, (model, param) in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    accuracy_model = cross_val_score(pipeline, X_train, y_train, cv=cv, n_jobs=-1, scoring='accuracy')
    mean_accuracy_model = np.mean(accuracy_model)
    print(colored(f'\n\n{name}:', 'red'))
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
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(colored(f"\n{name}:", 'red'))
    print(colored('Точность на тестах:', 'yellow'), f'{accuracy:.3F}')
    print(classification_report(y_test, y_pred))
    print(roc_auc_score(y_test, y_pred))
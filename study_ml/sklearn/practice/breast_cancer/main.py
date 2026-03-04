from sklearn.datasets import load_iris, load_wine, load_breast_cancer
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
from sklearn.metrics import classification_report, roc_auc_score

cv = KFold(n_splits=5, shuffle=True, random_state=42)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
# print(df.head())
# print(df.dtypes)
# print(df.isnull().sum())
X = df.drop('target', axis = 1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X.columns

models = {
    'Logistic Regression': (
        LogisticRegression(max_iter=20000),
        {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
    ),
    'Random Forest': (
        RandomForestClassifier(),
        {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(),
        {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5], 'subsample': [0.8, 1.0]}
    ),
    'Decision Tree': (
        DecisionTreeClassifier(),
        {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
    ),
    'SVM': (
        SVC(probability=True),
        {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    ),
    'K-Nearest Neighbors': (
        KNeighborsClassifier(),
        {'n_neighbors': [2, 3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
    )
}

best_models = {}

#Используем StandardScaler
print(colored("\n\n\t\t\tИспользуем StandardScaler:", 'blue'))
for name, (model, param) in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    accuracy_model = cross_val_score(pipeline, X_train, y_train, cv = cv, n_jobs=-1, scoring='accuracy')
    mean_accuracy_model = np.mean(accuracy_model)
    print(colored(f'\n\n{name}:', 'red'))
    print(colored('Точность до подбора гиперпараметров:', 'yellow'), f'{mean_accuracy_model:.3F}')
    grid = GridSearchCV(model, param, cv = cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    accuracy_best_model = cross_val_score(best_model, X_train, y_train, cv = cv, n_jobs=-1, scoring='accuracy')
    mean_accuracy_best_model = np.mean(accuracy_best_model)
    print(colored('Точность после подбора гиперпараметров:', 'yellow'), f'{mean_accuracy_best_model:.3F}')

#Не Используем StandardScaler
print(colored("\n\n\t\t\tНе используем StandardScaler:", 'blue'))
for name, (model, param) in models.items():
    accuracy_model = cross_val_score(model, X_train, y_train, cv = cv, n_jobs=-1, scoring='accuracy')
    mean_accuracy_model = np.mean(accuracy_model)
    print(colored(f'\n\n{name}:', 'red'))
    print(colored('Точность до подбора гиперпараметров:', 'yellow'), f'{mean_accuracy_model:.3F}')
    grid = GridSearchCV(model, param, cv = cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    accuracy_best_model = cross_val_score(best_model, X_train, y_train, cv = cv, n_jobs=-1, scoring='accuracy')
    mean_accuracy_best_model = np.mean(accuracy_best_model)
    print(colored('Точность после подбора гиперпараметров:', 'yellow'), f'{mean_accuracy_best_model:.3F}')

for name, model in best_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(colored(f"\n{name}:", 'red'))
    print(colored('Точность на тестах:', 'yellow'), f'{accuracy:.3F}')
    print(classification_report(y_test, y_pred))
    print(roc_auc_score(y_test, y_pred))







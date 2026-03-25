import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.metrics import accuracy_score        # Точность (Accuracy)
from sklearn.metrics import precision_score       # Precision
from sklearn.metrics import recall_score          # Recall
from sklearn.metrics import f1_score              # F1-score
from sklearn.metrics import roc_auc_score         # ROC-AUC
from sklearn.metrics import log_loss              # Log Loss
from sklearn.metrics import confusion_matrix      # Матрица ошибок
from termcolor import colored
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

models = {
    'LogisticRegression': LogisticRegression(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42)
}

df = pd.read_csv('animals_dataset.csv', sep=';')
X = df.drop(columns = 'is_pet_friendly', axis=1)
y = df['is_pet_friendly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X.select_dtypes(exclude = 'object').columns
cat_cols = X.select_dtypes(include = 'object').columns

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('\n', colored(name, 'red'))
    print(f'accuracy = {accuracy:.2f}')
    print(f'cv_mean = {cv_scores.mean():.2f}')  

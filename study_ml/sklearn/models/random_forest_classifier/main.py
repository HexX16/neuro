from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

#Признаки:
wine = load_wine()
features = np.array(wine.feature_names).reshape(-1,1)

# Загружаем данные
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Создаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказания
preds = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, preds)
print(f"Точность: {accuracy}")

# Важность признаков
importances = model.feature_importances_
for feature, impportance in sorted(zip(load_wine().feature_names, importances), key=lambda x: x[1], reverse=True)[:]:
    print(f"{feature}: {impportance:.3f}")
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy}")

importances = model.feature_importances_
for feature, importance in sorted(zip(load_wine().feature_names, importances), key = lambda x: x[1], reverse=True)[:5]:
    print(f"{feature}: {importance:.3f}")

# Визуализация дерева
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=load_wine().feature_names, class_names=load_wine().target_names, filled=True)
plt.show()

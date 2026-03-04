import pandas as pd
import numpy as np

# Твои исходные данные
data = {
    "technique": ["side", "post_toproll", "low_toproll", "hook", "triceps", "kingsmove"],
    "rise": [1, 5, 2, 1, 2, 5],
    "pronation": [3, 4, 5, 4, 1, 5],
    "supination": [3, 2, 2, 5, 5, 1],
    "wrist": [4, 4, 4, 5, 4, 2],
    "fingers": [4, 4, 4, 5, 4, 1],
    "brachioradialis": [3, 5, 5, 3, 3, 4],
    "biceps": [4, 3, 3, 4, 3, 3],
    "triceps": [4, 2, 2, 3, 5, 1],
    "side": [5, 2, 3, 5, 4, 2]
}

df = pd.DataFrame(data)

# Параметры генерации
num_copies = 30  # сколько экземпляров на каждую технику
noise_strength = 0.25  # насколько сильно можно варьировать признаки

augmented_rows = []

for _, row in df.iterrows():
    for i in range(num_copies):
        new_row = row.copy()
        for col in df.columns[1:]:  # все кроме 'technique'
            # Добавляем небольшое случайное изменение от -noise_strength до +noise_strength
            new_value = row[col] + np.random.uniform(-noise_strength, noise_strength)
            # Ограничиваем диапазон 1–5
            new_value = np.clip(new_value, 1, 5)
            new_row[col] = round(new_value, 2)
        augmented_rows.append(new_row)

augmented_df = pd.DataFrame(augmented_rows)

# Проверка результата
print(augmented_df.head(12))
print("\nКоличество строк:", len(augmented_df))

# Сохраним для модели
augmented_df.to_excel("tech_augmented_simple.xlsx", index=False)

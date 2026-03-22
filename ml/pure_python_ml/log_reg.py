import numpy as np
import pandas as pd
from models_utils import LogisticRegression as LogReg

np.random.seed(42)

age = np.random.randint(18, 60, 100)
income = np.random.randint(20000, 100000, 100)
time_on_site = np.random.randint(1, 60, 100)
x = np.column_stack((age, income, time_on_site))
y = ((x[:,1] > 50000) & (x[:,2] > 30)).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income,
    "time_on_site": time_on_site,
    "target": y
})

model = LogReg()
model.fit(x, y)
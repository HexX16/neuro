import matplotlib.pyplot as plt
import seaborn as sns
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
print(df)

# plt.hist(df['Score'])

# sns.scatterplot(data = df, x='Height', y='Weight', hue = 'Name')

# sns.boxplot(data = df, x = 'BMI', hue = 'Name')
# plt.title('boxplot')

plt.scatter(data=df, x = 'BMI', y='Score')

plt.show()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import seaborn as sns

df_wine = pd.read_csv('wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']

print('Class labels', np.unique(df_wine['Class label']))
  
df_wine.head()

print(df_wine)

# Step 3: Data Preprocessing
# Check for missing values
print(df_wine.isnull().sum())

# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df_wine.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# Feature and target separation
X = df_wine.drop(columns=['Class label'])
y = df_wine['Class label']

print(X)
print(y)


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lr = LogisticRegression(C=1.0,
                        penalty='l1',
                        random_state=42,
                        solver='liblinear',
                        multi_class='ovr')

lr.fit(X_train_scaled, y_train)

# X_combined_std = np.vstack((X_train_scaled, X_test_scaled))
# y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X_train_scaled,y_combined,classifier=lr)



# Calcular la precisión en el conjunto de prueba
accuracy = lr.score(X_test_scaled, y_test)
print('Precisión en el conjunto de prueba: {:.2f}'.format(accuracy))

# Calcular el error (1 - precisión)
error = 1 - accuracy
print('Error en el conjunto de prueba: {:.2f}'.format(error))

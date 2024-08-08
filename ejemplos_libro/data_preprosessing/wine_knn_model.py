import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import seaborn as sns

from sbs_sequence_backward_selection import SBS

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

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_scaled, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

# testing the features reduction
k3 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k3])


knn.fit(X_train_scaled, y_train)
print('Training accuracy:', knn.score(X_train_scaled, y_train))
print('Test accuracy:', knn.score(X_test_scaled, y_test))


knn.fit(X_train_scaled[:, k3], y_train)
print('Training accuracy:', knn.score(X_train_scaled[:, k3], y_train))
print('Test accuracy:', knn.score(X_test_scaled[:, k3], y_test))
#PCA (Principal Component analizys)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


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

# Feature and target separation
X = df_wine.drop(columns=['Class label'])
y = df_wine['Class label']

print(X)
print(y)


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot examples by class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

# initializing the PCA transformer and
# logistic regression estimator:
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')

# dimensionality reduction:
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# fitting the logistic regression model to the reduced dataset:
lr.fit(X_train_pca, y_train)

# plot decision regions:
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# Calcular la precisión en el conjunto de prueba
accuracy = lr.score(X_test_pca, y_test)
print('Precisión en el conjunto de prueba: {:.2f}'.format(accuracy))

# Calcular el error (1 - precisión)
error = 1 - accuracy
print('Error en el conjunto de prueba: {:.2f}'.format(error))
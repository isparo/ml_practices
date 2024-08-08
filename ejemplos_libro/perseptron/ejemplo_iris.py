import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.colors import ListedColormap

from perceptron import Perceptron


s = os.path.join('https://archive.ics.uci.edu', 'ml', 
                 'machine-learning-databases',
                 'iris','iris.data')

print('URL: ', s)

df = pd.read_csv(s, header=None, encoding='utf-8')
df.tail()

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# # Definir el conjunto de datos de entrada (X) y las etiquetas (y)
# Xin = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([-1, -1, -1, 1])

# # Crear una instancia de la clase Perceptron
ppn = Perceptron(eta=0.1, n_iter=50, random_state=1)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

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
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0],
                        y=X[y == cl, 1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=cl,
                        edgecolor='black')
            

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

print("Predicciones:", ppn.predict(np.array([[5.8,5.1]])))

# # Entrenar el modelo
# ppn.fit(Xin, y)

# # Imprimir los pesos ajustados y los errores durante el entrenamiento
# print("Pesos ajustados:", ppn.w_[1])
# print("Errores durante el entrenamiento:", ppn.errors_)
# print("Sesgo (bias) ajustado:", ppn.w_[0])


# print("Predicciones:", ppn.predict(np.array([[0.5,0.5]])))
# print("Predicciones:", ppn.predict(np.array([[0.5,20]])))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA

# Generamos un conjunto de datos no lineal
X, y = make_moons(n_samples=100, random_state=123)

# Visualizamos los datos originales
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Datos Originales")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()

# Aplicamos KPCA con los parámetros dados
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)

# Visualizamos los datos transformados
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
plt.title("Datos Transformados con KPCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()




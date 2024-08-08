import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
from matplotlib.colors import ListedColormap

class AdalineGD(object):
    """ADAptive LInear NEuron classifier.
       """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                                size=1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
       
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    


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

# norlization
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

# Crear una instancia de la clase Perceptron
adl = AdalineGD(eta=0.001, n_iter=15, random_state=1)

# Entrenar el modelo
adl.fit(X_std, y)

# Imprimir los pesos ajustados y los errores durante el entrenamiento
print("Pesos ajustados:", adl.w_[1])
print("cost_ durante el entrenamiento:", adl.cost_)
print("Sesgo (bias) ajustado:", adl.w_[0])

# Hacer predicciones
print("Predicciones:", adl.predict(X_std))

print("__")

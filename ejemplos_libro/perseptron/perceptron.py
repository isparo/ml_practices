import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
      
      def __init__(self, eta=0.01, n_iter=50, random_state=1):
          self.eta = eta
          self.n_iter = n_iter
          self.random_state = random_state

      def fit(self, X, y):
          rgen = np.random.RandomState(self.random_state)
          self.w_ = rgen.normal(loc=0.0, scale=0.01,
                                size=1 + X.shape[1])
          self.errors_ = []
          for _ in range(self.n_iter):
              errors = 0
              for xi, target in zip(X, y):
                  update = self.eta * (target - self.predict(xi))
                  self.w_[1:] += update * xi
                  self.w_[0] += update
                  errors += int(update != 0.0)
              self.errors_.append(errors)
          return self

      def net_input(self, X):
          return np.dot(X, self.w_[1:]) + self.w_[0]

      def predict(self, X):
          return np.where(self.net_input(X) >= 0.0, 1, -1)


# Definir el conjunto de datos de entrada (X) y las etiquetas (y)
Xin = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, -1, -1, 1])

# Crear una instancia de la clase Perceptron
ppn = Perceptron(eta=0.01, n_iter=50, random_state=1)

# Entrenar el modelo
ppn.fit(Xin, y)

# Imprimir los pesos ajustados y los errores durante el entrenamiento
print("Pesos ajustados:", ppn.w_[1])
print("Errores durante el entrenamiento:", ppn.errors_)
print("Sesgo (bias) ajustado:", ppn.w_[0])

# Hacer predicciones
print("Predicciones:", ppn.predict(Xin))


def plot_decision_boundary(perceptron, X, y):
    # Crear una malla de puntos
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Graficar la frontera de decisión y los puntos de datos
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Decision Boundary')
    plt.show()

# Usar la función de visualización
plot_decision_boundary(ppn, Xin, y)



# Unfortunately, the SBS algorithm has not been implemented in scikit-learn yet. 
# But since it is so simple, let's go ahead and implement it in Python from scratch

# SBS algorithm (Sequential Backward Selection)

# SBS (Sequential Backward Selection) es una técnica de selección de características
# utilizada para reducir la dimensionalidad de un conjunto de datos.
# El objetivo de SBS es identificar un subconjunto de características que maximicen
# el rendimiento de un modelo predictivo.

# Funcionamiento del SBS:
# 1. Inicialización:
#    Comienza con el conjunto completo de características.

# 2. Evaluación:
#    Evalúa el rendimiento del modelo usando todas las características.

# 3. Eliminación:
#    - Elimina una característica a la vez y evalúa el rendimiento del modelo sin esa característica.
#    - Selecciona la característica cuya eliminación causa la menor disminución (o la mayor mejora) en el rendimiento del modelo.

# 4. Repetición:
#    Repite los pasos 2 y 3 hasta que se alcance un criterio de parada, como un número deseado de características
#    o una mejora mínima en el rendimiento.

# Resultado:
#    El resultado es un subconjunto de características seleccionadas que proporciona un buen equilibrio
#    entre el rendimiento del modelo y la simplicidad.

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS():
    def __init__(self, estimator,
                 k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=self.test_size,
                         random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,X_test, y_test, self.indices_)
        
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                        X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
                
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
    
        return self

    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        
        return score

        

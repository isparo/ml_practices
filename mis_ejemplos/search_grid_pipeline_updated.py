import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
url = 'fetal_health.csv'
df = pd.read_csv(url)

# Separar características y la variable objetivo
X = df.drop('fetal_health', axis=1).values
y = df['fetal_health'].values

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Crear el pipeline con los mejores parámetros
pipe = Pipeline([
    ('scaler', StandardScaler()),           # Estandarizar los datos
    #('pca', PCA(n_components=2)),           # Reducir a 2 dimensiones con PCA
    ('svc', SVC(C=10, kernel='rbf'))         # SVM con los mejores parámetros encontrados
])

# Entrenar el modelo con los datos de entrenamiento
pipe.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = pipe.predict(X_test)
print("\nReporte de clasificación en el conjunto de prueba:")
print(classification_report(y_test, y_pred))

# Calcular y mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(conf_matrix)

# Visualizar la matriz de confusión con un heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Clase 1', 'Clase 2', 'Clase 3'],
            yticklabels=['Clase 1', 'Clase 2', 'Clase 3'])
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.title('Matriz de Confusión')
plt.show()

# Validación cruzada simple usando cross_val_score para evaluar la consistencia
scores = cross_val_score(pipe, X_train, y_train, cv=10)
print("\nPuntajes de validación cruzada:")
print(scores)
print("Promedio de validación cruzada:", np.mean(scores))

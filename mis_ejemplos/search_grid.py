import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
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

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Definir el modelo SVM
svc = SVC(random_state=42)

# Definir los parámetros para la búsqueda en grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Configurar la búsqueda en grid con validación cruzada
# cv=5, scoring='accuracy', n_jobs=-1
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Ajustar el modelo usando la validación cruzada y búsqueda en grid
grid_search.fit(X_train_pca, y_train)

# Imprimir los mejores parámetros encontrados por GridSearchCV
print("Mejores parámetros encontrados por GridSearchCV:")
print(grid_search.best_params_)

# Evaluar el modelo en el conjunto de prueba
y_pred = grid_search.predict(X_test_pca)
print("\nReporte de clasificación en el conjunto de prueba:")
print(classification_report(y_test, y_pred))


# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión
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


# Validación cruzada simple usando cross_val_score
scores = cross_val_score(grid_search.best_estimator_, X_train_pca, y_train, cv=10)
print("\nPuntajes de validación cruzada:")
print(scores)
print("Promedio de validación cruzada:", np.mean(scores))

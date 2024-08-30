import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import LabelEncoder

cancerDS = pd.read_csv('./dataset.csv')
cancerDS.head()
cancerDS.info()
cancerDS.columns

y = cancerDS['diagnosis']
X = cancerDS.drop(['id','diagnosis','Unnamed: 32'],axis=1)

le = LabelEncoder()
y = le.fit_transform(y)
le.transform(['M', 'B'])


# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Encontrar los mejores hiperparámetros para DecisionTreeClassifier
param_grid_dt = {
    'max_depth': [1, 2, 3, None],
    'criterion': ['gini', 'entropy']
}

grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=10, scoring='accuracy')
grid_search_dt.fit(X_train_scaled, y_train)

# Obtener el mejor clasificador base
best_decision_tree = grid_search_dt.best_estimator_

print("Mejores parámetros encontrados para DecisionTreeClassifier:")
print(grid_search_dt.best_params_)

# Crear el BaggingClassifier
bagging_clf = BaggingClassifier(
    estimator=best_decision_tree,  # Clasificador base
    random_state=42
)

# Definir el grid de hiperparámetros para el BaggingClassifier
param_grid = {
    'n_estimators': [10, 20, 30],  # Número de clasificadores en el ensamblado
    'max_samples': [0.5, 0.8, 1.0],  # Fracción de muestras para entrenar cada clasificador
    'max_features': [0.5, 0.8, 1.0]  # Fracción de características para entrenar cada clasificador
}

# Configurar la búsqueda en grid con validación cruzada
grid_search = GridSearchCV(bagging_clf, param_grid, cv=10, scoring='accuracy', n_jobs=-1)

# Ajustar el modelo usando la validación cruzada y búsqueda en grid
grid_search.fit(X_train_scaled, y_train)

# Mejor modelo y parámetros
best_bagging_clf = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Mejores parámetros encontrados por GridSearchCV:")
print(best_params)

# Evaluar el mejor modelo en el conjunto de prueba
y_pred = best_bagging_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy en el conjunto de prueba:")
print(accuracy)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

print('visualize over and under fitting')


# Evaluar en el conjunto de entrenamiento
y_train_pred = best_bagging_clf.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Evaluar en el conjunto de prueba
y_test_pred = best_bagging_clf.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))


#matriz de confucion
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print('matriz confucion train data: \n')
print(conf_matrix_train)

conf_matrix_test = confusion_matrix(y_test, y_pred)
print('matriz confucion test data: \n')
print(conf_matrix_test)

# # Determinar si hay overfitting o underfitting
# if train_accuracy > test_accuracy:
#     print("El modelo podría estar sufriendo de overfitting.")
# elif train_accuracy < test_accuracy:
#     print("El modelo podría estar sufriendo de underfitting.")
# else:
#     print("El modelo tiene un buen equilibrio entre el conjunto de entrenamiento y prueba.")
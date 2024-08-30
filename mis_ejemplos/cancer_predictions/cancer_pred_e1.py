import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA


import seaborn as sns
import matplotlib.pyplot as plt

cancerDS = pd.read_csv('./dataset.csv')
cancerDS.head()
cancerDS.info()
cancerDS.columns

y = cancerDS['diagnosis']
X = cancerDS.drop(['id','diagnosis','Unnamed: 32'],axis=1)

le = LabelEncoder()
y = le.fit_transform(y)
le.transform(['M', 'B'])

print(y)


# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar un modelo SVM
#model = SVC(random_state=42, C=0.1, kernel='linear')

# {'C': 100, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'
model = LogisticRegression(random_state=42, max_iter=5000, C=100, solver='liblinear')


# entre el modelo
model.fit(X_train_scaled, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test_scaled)


# Evaluar el modelo
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Mostrar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benigno', 'Maligno'], 
            yticklabels=['Benigno', 'Maligno'])
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.title('Matriz de Confusión')
plt.show()


print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# # Definir la cuadrícula de hiperparámetros
# param_grid = {
#    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverso de la regularización
#     'penalty': ['l1', 'l2'],                # Tipo de penalización: L1 o L2
#     'solver': ['liblinear', 'newton-cg', 'lbfgs', 'saga'],  # Algoritmo de optimización
#     'max_iter': [100, 200, 300]             # Número máximo de iteraciones para el algoritmo de optimización
# }


# # Configurar la búsqueda en grid con validación cruzada
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# # Ajustar el modelo usando la validación cruzada y búsqueda en grid
# grid_search.fit(X_train, y_train)

# # Imprimir los mejores parámetros encontrados por GridSearchCV
# print("Mejores parámetros encontrados por GridSearchCV:")
# print(grid_search.best_params_)

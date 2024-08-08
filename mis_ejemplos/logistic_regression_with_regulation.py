import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
import numpy as np

# Paso 1: Cargar los datos
df = pd.read_csv('fetal_health.csv')

# Paso 2: Preprocesar los datos
# Separar características y etiquetas
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Paso 3: Crear el modelo con los hiperparámetros especificados
model = LogisticRegression(
    C=1.0,
    random_state=1,
    solver='lbfgs',
    multi_class='ovr'
)

# Entrenar el modelo
model.fit(X_train_std, y_train)

# Paso 4: Realizar predicciones y evaluar el modelo
y_pred = model.predict(X_test_std)

# Evaluar el rendimiento
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

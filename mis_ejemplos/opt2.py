import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# Aplicar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Paso 3: Crear el modelo con los hiperparámetros especificados
model = LogisticRegression(
    C=1.0,
    random_state=1,
    solver='lbfgs',
    
    multi_class='multinomial'
)

# Entrenar el modelo
model.fit(X_train_pca, y_train)

# Paso 4: Realizar predicciones y evaluar el modelo
y_pred = model.predict(X_test_pca)

# Evaluar el rendimiento
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Paso 5: Visualizar la frontera de decisión
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA']))
plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1],
            c='red', marker='o', label='Normal')
plt.scatter(X_train_pca[y_train == 2, 0], X_train_pca[y_train == 2, 1],
            c='blue', marker='x', label='Suspect')
plt.scatter(X_train_pca[y_train == 3, 0], X_train_pca[y_train == 3, 1],
            c='green', marker='s', label='Pathological')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(loc='best')
plt.title('Frontera de Decisión y Datos de Entrenamiento')
plt.show()

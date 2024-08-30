#this first example uses the Iris dataset
# • Logistic regression classifier
# • Decision tree classifier
# • k-nearest neighbors classifier


# Ejemplo Conceptual:
# Supongamos que tienes tres modelos y los evalúas en términos de precisión:
# • Modelo 1 (Logistic Regression): Precisión = 0.80
# • Modelo 2 (SVC): Precisión = 0.85
# • Modelo 3 (Decision Tree): Precisión = 0.75

# Podrías asignar los pesos de manera proporcional a estas precisiones. Por ejemplo:
# • Peso del Modelo 1 = 0.80 / (0.80 + 0.85 + 0.75) ≈ 0.33
# • Peso del Modelo 2 = 0.85 / (0.80 + 0.85 + 0.75) ≈ 0.35
# • Peso del Modelo 3 = 0.75 / (0.80 + 0.85 + 0.75) ≈ 0.31


from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.5,
                                                    random_state=1,
                                                    stratify=y)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf1 = LogisticRegression(penalty='l2',
                          C=100,
                          max_iter=500,
                          solver='liblinear',
                          random_state=42)

clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=42)

clf3 = KNeighborsClassifier(n_neighbors=5,
                            p=2,
                            metric='minkowski')


# Entrenar los modelos base
clf1.fit(X_train_scaled, y_train)
clf2.fit(X_train_scaled, y_train)
clf3.fit(X_train_scaled, y_train)

# Evaluar los modelos y calcular la exactitud
accuracy1 = accuracy_score(y_test, clf1.predict(X_test_scaled))
accuracy2 = accuracy_score(y_test, clf2.predict(X_test_scaled))
accuracy3 = accuracy_score(y_test, clf3.predict(X_test_scaled))

print(f'Accuracy of Logistic Regression: {accuracy1:.2f}')
print(f'Accuracy of Decision Tree: {accuracy2:.2f}')
print(f'Accuracy of KNN: {accuracy3:.2f}')


# Crear el VotingClassifier (hard voting)
voting_clf_hard = VotingClassifier(estimators=[('lr', clf1), 
                                          ('dt', clf2), 
                                          ('kn', clf3)],
                              voting='hard')

# Asignar pesos basados en la exactitud
#weightsAccu = [accuracy1, accuracy2, accuracy3]

# Precisión de cada modelo
precisions = np.array([accuracy1, accuracy2, accuracy3])

# Calcular pesos proporcionales a las precisiones
weights = precisions / np.sum(precisions)

# Si prefieres tener pesos enteros, puedes multiplicar por un factor (ej. 100)
integer_weights = (weights * 100).astype(int)

print("Pesos proporcionales:", weights)
print("Pesos enteros:", integer_weights)


voting_clf_soft = VotingClassifier(estimators=[('lr', clf1), 
                                          ('dt', clf2), 
                                          ('kn', clf3)],
                                   weights=integer_weights,
                                   voting='soft')

# Entrenar y evaluar
voting_clf_hard.fit(X_train_scaled, y_train)
print("Accuracy (hard): ", voting_clf_hard.score(X_test_scaled, y_test))

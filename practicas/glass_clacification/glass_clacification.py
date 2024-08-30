"""
    1) cargar el dataset √
        - standarizar √
        - normalizar (si es necesarío)√
    2) buscar el modelo perfecto √
        - plot clacificaciones √
        - searcg_grid
        - matriz de coorelacion √
        - 
    4) clacificar los datos
    
    6) ver si se requiere PCA, LDA, KPCA
    

"""

from random import shuffle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

# URL del dataset de Glass Identification en la UCI Repository
url = "./dataset/glass.data"

# Nombres de las columnas basados en la documentación del dataset
columns = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]

# Cargar el dataset
glass_df = pd.read_csv(url, names=columns, index_col=False)

y = glass_df['Type']
X = glass_df.drop(['Id', 'Type'], axis=1)


# Supongamos que X e y son tus datos y etiquetas.
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

print(X_resampled)
print(y_resampled)

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2,  random_state=1)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# grafico de conteo de claces
plt.figure(figsize=(6, 4))
sns.countplot(x='Type', data=glass_df)
plt.title('Distribution of Glass Classes')
#plt.show()

# Matriz de coorelacion
plt.figure(figsize=(12, 10))
sns.heatmap(glass_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix') 

# plt.show()

# TODO: Implementar un modelo de clasificacion y evaluar su rendimiento

models = {
    'Random Forest': RandomForestClassifier( random_state=1)
}

param_grids = {
    'Random Forest':{
        'n_estimators': [50,100, 200, 300], # Número de árboles en el bosque
        'max_depth': [5,10, 20,30, None], # Profundidad máxima de los árboles
        # 'min_samples_split': [2, 5, 10], # Número mínimo de muestras necesarias para dividir un nodo
        # 'min_samples_leaf': [1, 2, 4] # Número mínimo de muestras necesarias en una hoja
        # 'max_features': ['auto', 'sqrt', 'log2'] # Número de características a considerar al buscar la mejor división
        # 'bootstrap': [True, False] # Método de muestreo para construir los árboles (con o sin reemplazo)
    }
}

print('searching best models')
for name, model in models.items():
    print('___________________________________________')
    
    print(f'analyzing {name}')
    model = model.fit(X_train_scaled, y_train)
    
    print(f'\n')
   
    y_test_pred = model.predict(X_test_scaled)
    y_train_pred = model.predict(X_train_scaled)
    print('training report')
    print(classification_report(y_train, y_train_pred))
    print('test report')
    print(classification_report(y_test, y_test_pred))
    
    print('varianza')
    model_train = accuracy_score(y_train, y_train_pred)
    model_test = accuracy_score(y_test, y_test_pred)

    print('train/test accuracies %.3f/%.3f' % (model_train, model_test))
    
    print('******** Evaluation whit search_grid**********')
    
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    # best_models[name] = grid_search.best_estimator_
    print(f'Best parameters for {name}: {grid_search.best_params_}')
    
    best_model = grid_search.best_estimator_
    y_best_pred = best_model.predict(X_test_scaled)
    print(f'{name} Classification Report:\n')
    print(classification_report(y_test, y_best_pred))
    print('accuracy_score = %.3f' % accuracy_score(y_test, y_best_pred))
    
    print(confusion_matrix(y_test, y_best_pred))
    
    # ada = AdaBoostClassifier(estimator=best_model,
    #                      n_estimators=500,
    #                      learning_rate=0.1,
    #                      random_state=1)
    
    # ada = ada.fit(X_train, y_train)
    # y_train_pred_ada = ada.predict(X_train)
    # y_test_pred_ada = ada.predict(X_test)

    # ada_train = accuracy_score(y_train, y_train_pred_ada)
    # ada_test = accuracy_score(y_test, y_test_pred_ada)
    # print('AdaBoost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))

    
    # print('implementing bagging algorithm')
    # bagging = BaggingClassifier(estimator=best_model,
    #                          n_estimators=500,
    #                          max_samples=1.0,
    #                          max_features=1.0,
    #                          bootstrap=True,
    #                          bootstrap_features=False,
    #                          n_jobs=1,
    #                          random_state=1)
    

    # bagging = bagging.fit(X_train, y_train)
    # y_train_pred_bag = bagging.predict(X_train)
    # y_test_pred_bag = bagging.predict(X_test)

    # bagging_train = accuracy_score(y_train, y_train_pred_bag)
    # bagging_test = accuracy_score(y_test, y_test_pred_bag)

    # print('Bagging train/test accuracies %.3f/%.3f' % (bagging_train, bagging_test))
    
    print('___________________________________________')
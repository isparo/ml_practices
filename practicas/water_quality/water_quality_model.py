import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import resample
from skopt.space import Real, Categorical, Integer

from imblearn.over_sampling import SMOTE

from skopt import BayesSearchCV


# Paso 1: Cargar los datos
df = pd.read_csv('./dataset/water_potability.csv')

# Crear el imputador
imputer = SimpleImputer(strategy='mean')  # También puedes usar 'median' o 'most_frequent'
# Aplicar el imputador y convertir de nuevo a DataFrame
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print(df)

# Paso 2: Preprocesar los datos
# Separar características y etiquetas
X = df.drop('Potability', axis=1)
y = df['Potability']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,  random_state=17)

# Normalizar los datos
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Paso 3: Crear el modelo con los hiperparámetros especificados
# model = LogisticRegression(
#     C=1.0,
#     random_state=1,
#     solver='lbfgs'
#     # multi_class='ovr'
# )

model = RandomForestClassifier( random_state=17)

model.fit(X_train_std, y_train)
y_train_pred = model.predict(X_train_std)
y_test_pred = model.predict(X_test_std)

print('training report')
print(classification_report(y_train, y_train_pred))
print('test report')
print(classification_report(y_test, y_test_pred))

print('varianza')
model_train = accuracy_score(y_train, y_train_pred)
model_test = accuracy_score(y_test, y_test_pred)

print('train/test accuracies %.3f/%.3f' % (model_train, model_test))


################################################################

df_majority = df[df.Potability==0]
df_minority = df[df.Potability==1]

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                  random_state=17)

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

X_upsampled = df_upsampled.drop('Potability', axis=1)
y_upsampled = df_upsampled['Potability']

X_train_up, X_test_up, y_train_up, y_test_up = train_test_split(X_upsampled, y_upsampled, test_size=0.2,  random_state=17)

# Normalizar los datos
scalerUp = StandardScaler()
X_train_std_up = scalerUp.fit_transform(X_train_up)
X_test_std_up = scalerUp.transform(X_test_up)


search_space = {
    'n_estimators': Integer(10, 200),
    'max_depth': Integer(1, 20),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 5),
    'criterion': Categorical(['gini', 'entropy'])#'log_loss'
}

bayes_search = BayesSearchCV(
    model,
    search_space,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=2,
     random_state=17
)

bayes_search.fit(X_train_std_up, y_train_up)

best_model = bayes_search.best_estimator_

y_pred_bs = best_model.predict(X_test_std_up)

accuracy = accuracy_score(y_test_up, y_pred_bs)
print(f"Accuracy: {accuracy:.3f}")

print("Best hyperparameters:", bayes_search.best_params_)
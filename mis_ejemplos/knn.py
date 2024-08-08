import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sbs_sequence_backward_selection import SBS

df = pd.read_csv('fetal_health.csv')

df.head()

# Step 3: Data Preprocessing
# Check for missing values
print(df.isnull().sum())

# Descriptive statistics
print(df.describe())

# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Feature and target separation
selected_columns = ['accelerations','prolongued_decelerations', 'abnormal_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability']
X = df.drop(columns=['fetal_health']) #df[selected_columns]
y = df['fetal_health']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

knn.fit(X_train_scaled, y_train)

# Paso 4: Realizar predicciones y evaluar el modelo
y_pred = knn.predict(X_test_scaled)

# Evaluar el rendimiento
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)


print('----------------------------------------------------------------')

# Predicción para un nuevo dato
# new_data = pd.DataFrame([[129.0,0.0,0.001,0.006,0.006,0.0,0.003,66.0,2.9,0.0,0.0,94.0,50.0,144.0,8.0,0.0,105.0,85.0,109.0,11.0,0.0]], 
#                         columns=X.columns)

# new_data_scaled = scaler.transform(new_data)
# prediction = knn.predict(new_data_scaled)

# print('Prediction for new data:', prediction[0])

print('----------------------------------------------------------------')

# sbs = SBS(knn, k_features=1)
# sbs.fit(X_train_scaled, y_train)

# k_feat = [len(k) for k in sbs.subsets_]
# plt.plot(k_feat, sbs.scores_, marker='o')
# plt.ylim([0.7, 1.02])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
# plt.tight_layout()
# plt.show()
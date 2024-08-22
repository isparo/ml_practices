import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('fetal_health.csv')

# Ver los nombres de las columnas para saber cuáles usar
print(df.columns)

# Gráfico de dispersión entre 'baseline value' y 'accelerations'
sns.scatterplot(x='baseline value', y='accelerations', hue='fetal_health', data=df)
plt.title("Gráfico de Dispersión de Dos Características")
plt.xlabel("Baseline Value")
plt.ylabel("Accelerations")
plt.show()

sns.countplot(x='fetal_health', data=df)
plt.title("Distribución de Clases en el Dataset")
plt.show()
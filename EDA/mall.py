import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
df = pd.read_csv("Mall_Customers.csv")

# Ajustes visuales
sns.set(style="whitegrid")

# Información general
print("Información general del dataset:")
print(df.info())
print("\nResumen estadístico:")
print(df.describe())

# Verificar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Histogramas de variables numéricas
plt.figure(figsize=(15, 4))
for i, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribución de {col}')
plt.tight_layout()
plt.show()

# Boxplots para detección de outliers
plt.figure(figsize=(15, 4))
for i, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(data=df, y=col)
    plt.title(f'Boxplot de {col}')
plt.tight_layout()
plt.show()

# Distribución por género
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x='Gender')
plt.title('Distribución por Género')
plt.show()

# Mapa de calor de correlaciones
plt.figure(figsize=(6, 4))
sns.heatmap(df.drop(columns='CustomerID').corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor de Correlaciones')
plt.show()

# Relación Ingreso Anual vs Spending Score
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.title('Ingreso Anual vs Spending Score')
plt.show()

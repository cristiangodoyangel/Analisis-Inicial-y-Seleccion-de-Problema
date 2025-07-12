# EDA - Netflix Movies and TV Shows Dataset

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Cargar el dataset
df = pd.read_csv('netflix_titles.csv')  # Asegúrate de tener el archivo en la misma carpeta
df.head()

# Información general del dataset
print(df.info())
print(df.describe(include='all'))

# Revisión de valores nulos
print("Valores nulos por columna:")
print(df.isnull().sum())

print("\nPorcentaje de valores nulos:")
print((df.isnull().sum() / len(df)) * 100)

# Limpieza básica de datos
df.drop_duplicates(inplace=True)

df['director'].fillna('Desconocido', inplace=True)
df['cast'].fillna('Desconocido', inplace=True)
df['rating'].fillna('Desconocido', inplace=True)
df['country'].fillna('Desconocido', inplace=True)
df['date_added'].fillna('No disponible', inplace=True)

# Conteo por tipo de contenido
sns.countplot(data=df, x='type', palette='pastel')
plt.title('Distribución de tipo de contenido en Netflix')
plt.xlabel('Tipo de contenido')
plt.ylabel('Cantidad')
plt.show()

# Top 10 países con más títulos
top_paises = df['country'].value_counts().head(10)
sns.barplot(x=top_paises.values, y=top_paises.index)
plt.title('Top 10 países con más títulos en Netflix')
plt.xlabel('Cantidad de títulos')
plt.ylabel('País')
plt.show()

# Títulos por año de lanzamiento
df['release_year'].value_counts().sort_index().plot(kind='bar', figsize=(16,6))
plt.title('Cantidad de títulos lanzados por año')
plt.xlabel('Año')
plt.ylabel('Cantidad de títulos')
plt.show()

# Duración de películas
peliculas = df[df['type'] == 'Movie'].copy()
peliculas['duration_mins'] = peliculas['duration'].str.extract('(\d+)').astype(float)

sns.histplot(peliculas['duration_mins'], bins=30, kde=True)
plt.title('Duración de Películas en Netflix')
plt.xlabel('Duración (minutos)')
plt.ylabel('Cantidad')
plt.show()

# Géneros más comunes
generos = df['listed_in'].str.split(', ', expand=True).stack().value_counts().head(10)
sns.barplot(y=generos.index, x=generos.values)
plt.title('Top 10 géneros más frecuentes en Netflix')
plt.xlabel('Cantidad')
plt.ylabel('Género')
plt.show()

# Mapa de calor de correlación (solo duración numérica)
sns.heatmap(peliculas[['duration_mins']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlación entre duración de películas')
plt.show()

# EDA terminado, no veo un problema real para analizar y predecir

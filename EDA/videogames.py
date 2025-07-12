# EDA - Video Game Sales (vgsales.csv)

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Carga del dataset
df = pd.read_csv('vgsales.csv')  # Asegúrate de tener el archivo en el mismo directorio
df.head()

# Información general
print("Información del dataset:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe(include='all'))

# Valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Limpieza de datos
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)  # Si no querés eliminar, podés usar fillna()

# Top 10 géneros más comunes
sns.countplot(data=df, y='Genre', order=df['Genre'].value_counts().index[:10], palette='muted')
plt.title('Top 10 géneros de videojuegos')
plt.xlabel('Cantidad de juegos')
plt.ylabel('Género')
plt.show()

# Ventas globales por género
ventas_genero = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
sns.barplot(x=ventas_genero.values, y=ventas_genero.index, palette='viridis')
plt.title('Ventas globales por género')
plt.xlabel('Ventas en millones')
plt.ylabel('Género')
plt.show()

# Top 10 plataformas con más juegos
top_platforms = df['Platform'].value_counts().head(10)
sns.barplot(x=top_platforms.values, y=top_platforms.index)
plt.title('Top 10 plataformas con más juegos')
plt.xlabel('Cantidad')
plt.ylabel('Plataforma')
plt.show()

# Juegos por año
df['Year'] = df['Year'].astype(int)  # Asegurar que sea numérico
df['Year'].value_counts().sort_index().plot(kind='line')
plt.title('Lanzamientos de videojuegos por año')
plt.xlabel('Año')
plt.ylabel('Cantidad de juegos')
plt.grid(True)
plt.show()

# Ventas globales por año
ventas_anuales = df.groupby('Year')['Global_Sales'].sum().sort_index()
ventas_anuales.plot(kind='bar', figsize=(16,6))
plt.title('Ventas globales de videojuegos por año')
plt.xlabel('Año')
plt.ylabel('Ventas (millones)')
plt.show()

# Mapa de calor de correlaciones entre regiones
ventas_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
sns.heatmap(df[ventas_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlación entre ventas por región')
plt.show()


# avanzaré con los demás datasets para elegir

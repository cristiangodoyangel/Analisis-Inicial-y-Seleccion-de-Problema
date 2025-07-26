# Análisis Inicial y Selección de Problema

📌 Descripción del Proyecto
Este proyecto tiene como objetivo realizar un análisis exploratorio de datos (EDA) de cuatro datasets diversos y seleccionar una problemática de Machine Learning para abordar. El análisis cubre datos del entretenimiento, videojuegos, comportamiento de clientes y fenómenos naturales.

# 📂 Conjuntos de Datos Analizados

## Netflix Titles

- Películas y series en la plataforma.

- Variables: tipo, duración, país, año, género, etc.

## Ventas de Videojuegos

- Títulos por plataforma, género y ventas por región.

## Mall Customers

- Datos de clientes: edad, ingresos y puntaje de gasto.

## Terremotos en Chile (Dataset Elegido)

- Registros sísmicos con fecha, magnitud, latitud y longitud.

Fuente: Kaggle

# 📊 Resumen del EDA

Se identificaron valores nulos, rangos atípicos y tendencias temporales.

Se aplicaron visualizaciones como histogramas, scatter plots y mapas de calor.

Se analizaron patrones de distribución en espacio y tiempo.

# 🧩 Problema Seleccionado

Tipo de problema: Predicción (Regresión)

Dataset elegido: Terremotos en Chile

## Justificación:

Dado el contexto geológico del país, prever la magnitud de sismos futuros o identificar zonas de mayor riesgo es una problemática relevante. Se buscará aplicar modelos de regresión para predecir la magnitud de un sismo basado en ubicación y características temporales.

Objetivo específico:
Predecir la magnitud de un sismo a partir de variables como profundidad, latitud, longitud, fecha y hora.

# ▶️ Instrucciones para Ejecutar

Clonar el repositorio:

´´´
git clone https://github.com/[tu_usuario]/Prediccion-Machine-Learning
´´´

Instalar dependencias:

´´´
pip install pandas matplotlib seaborn
´´´
# Ejecutar los notebooks o scripts ubicados en la carpeta /EDA.

#👥 Autor
Cristian Godoy Ángel

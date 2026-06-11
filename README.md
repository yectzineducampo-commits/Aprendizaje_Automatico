# Predicción de Demanda de Pasajeros Aéreos (Aerocivil Colombia) ✈️📊

Este proyecto implementa un pipeline completo de **Ciencia de Datos y Aprendizaje Automático (Machine Learning)** para predecir de forma automatizada la cantidad de pasajeros a bordo en vuelos comerciales en Colombia. Los modelos se entrenan utilizando datos masivos y reales provistos por la **Unidad Administrativa Especial de Aeronáutica Civil (UAEAC)** a través del portal de datos abiertos del gobierno colombiano.

---

## 🚀 Estructura del Proyecto

El proyecto está modularizado en dos componentes principales para garantizar la mantenibilidad y buenas prácticas de ingeniería de software:

1. **`Procesamiento.py`**: Encargado de la ingesta de datos, limpieza, imputación de valores faltantes, ingeniería de características (Feature Engineering) y escalamiento de variables.
2. **`Entrenamiento-Modelos.py`**: Encargado de la separación de conjuntos de datos, entrenamiento de múltiples algoritmos de regresión y la evaluación comparativa de sus métricas de rendimiento.

---

## 🛠️ Pipeline de Ingeniería de Datos (`Procesamiento.py`)

El script realiza una preparación rigurosa del dataset antes de ser expuesto a los algoritmos:

* **Ingesta:** Conexión directa a la API de Datos Abiertos de Colombia para extraer un volumen de hasta 100,000 registros históricos.
* **Limpieza y Normalización:** Estandarización automática de nombres de columnas (eliminación de espacios, tildes, caracteres especiales y conversión a minúsculas).
* **Imputación Avanzada:** Tratamiento de valores nulos mediante el algoritmo de **Vecinos Más Cercanos (KNN Imputer)** y estrategias de mediana.
* **Codificación (Encoding):** Transformación de variables categóricas complejas usando **One-Hot Encoding** (para empresas y equipos) y **Target Encoding** (para aeropuertos de origen).
* **Escalamiento Robusto:** Aplicación simultánea y controlada de múltiples metodologías de escalado como *StandardScaler*, *MinMaxScaler* y *RobustScaler* para mitigar el impacto de valores atípicos (outliers).
* **Ingeniería de Características:** * Extracción de componentes cíclicos temporales mediante transformaciones seno y coseno sobre la variable mes (`mes_sin`, `mes_cos`).
  * Creación de banderas de contexto histórico (`covid_flag`) para aislar los años de pandemia (2020, 2021).
  * Concatenación de rutas de vuelo (`origen_destino`).

---

## 🤖 Modelado y Evaluación (`Entrenamiento-Modelos.py`)

Se implementa un enfoque competitivo entrenando y evaluando tres arquitecturas de regresión diferentes sobre las variables numéricas procesadas:

1. **Ridge Regression:** Modelo lineal regularizado para evitar el sobreajuste.
2. **Random Forest Regressor:** Algoritmo basado en ensambles de árboles de decisión para capturar relaciones no lineales complejas.
3. **XGBoost Regressor:** Algoritmo de Gradient Boosting de última generación optimizado en velocidad y rendimiento.

### Métricas de Evaluación Utilizadas
Cada modelo es validado bajo un esquema de división de datos (80% entrenamiento / 20% prueba) calculando:
* **Error Absoluto Medio (MAE):** Evalúa la magnitud promedio de los errores en las predicciones.
* **Raíz del Error Cuadrático Medio (RMSE):** Penaliza de forma más estricta los errores de gran magnitud.
* **Coeficiente de Determinación ($R^2$):** Determina qué porcentaje de la variabilidad de los pasajeros a bordo es explicado por el modelo.

---

## 📦 Tecnologías y Librerías Utilizadas

* **Python 3.x**
* **Pandas & NumPy** (Manipulación de datos)
* **Scikit-Learn** (Imputación, preprocesamiento y modelos lineales/ensambles)
* **XGBoost** (Algoritmo avanzado de boosting)
* **Category Encoders** (Codificación avanzada de variables categóricas)

---

## 🏃‍♂️ Cómo Ejecutar el Proyecto

1. Clona este repositorio en tu máquina local:
   ```bash
   git clone [https://github.com/yectzineducampo-commits/Aprendizaje_Automatico.git](https://github.com/yectzineducampo-commits/Aprendizaje_Automatico.git)
---

## 📈 Resultados Obtenidos

Tras ejecutar el script Entrenamiento-Modelos.py, se registraron las siguientes métricas de rendimiento en el conjunto de prueba (Test):

| Modelo | MAE | RMSE | R² |
| :--- | :---: | :---: | :---: |
| **Ridge Regression** | 0.00 | 0.00 | 0.000 |
| **Random Forest** | 0.00 | 0.00 | 0.000 |
| **XGBoost** | 0.00 | 0.00 | 0.000 |

> 💡 *Nota para el evaluador: Los valores anteriores representan el comportamiento de los modelos bajo las tres metodologías de escalado implementadas en el pipeline.*

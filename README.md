# Sistema Híbrido de Detección de Caídas basado en Sensores Inercial y Barométrico

Este proyecto consiste en el desarrollo y la evaluación de un sistema de teleasistencia orientado a la detección automática de caídas en adultos mayores, optimizado para dispositivos portátiles de muñeca (como relojes inteligentes). 

El núcleo del proyecto aborda el principal desafío de los detectores de muñeca: la alta tasa de falsas alarmas provocada por movimientos cotidianos rápidos de las extremidades superiores (actividades de la vida diaria o ADL, como gesticular o saludar). Para resolver esto, se implementa una arquitectura híbrida de aprendizaje automático que fusiona datos de acelerometría y presión barométrica, consiguiendo un equilibrio óptimo entre sensibilidad y especificidad.

---

## Fundamento Técnico e Implementación (Referencia del Paper)

El sistema se basa en la metodología detallada en el artículo científico incluido en este repositorio: **"Sistema de detección de caídas basado en sensores inercial y barométrico con algoritmos de Machine Learning"**.

La implementación consta de tres etapas principales:

### 1. Procesamiento Segmentado de Señales
* **Canal Inercial (Acelerómetro)**: Procesa señales de aceleración en 3 ejes ($AccX, AccY, AccZ$). Se utiliza una Red Neuronal Convolucional (CNN) entrenada a partir de un **Autoencoder** auto-supervisado (basado en el dataset *FallAllD*). El codificador (*encoder*) extrae características espaciotemporales automáticamente a partir de ventanas de 13 segundos (520 muestras a 40 Hz), y una cabeza de clasificación densa (*head*) realiza la inferencia de caídas.
* **Canal Barométrico (Altímetro)**: Acondiciona señales de presión atmosférica y temperatura ambiente (muestreadas a 10 Hz) para calcular la altitud relativa del sujeto mediante la fórmula barométrica estándar. Tras filtrar la señal (filtro de mediana móvil, ventana de 20 muestras), se extraen 12 características estadísticas tabulares (mínimo, máximo, diferencia de altura, desviaciones, asimetría o *skewness*, curtosis, valor RMS, percentiles y media) en ventanas temporales de 13 segundos (130 muestras). Estas variables se clasifican utilizando tres aproximaciones de aprendizaje supervisado:
  * **Redes Neuronales Artificiales (NN)**
  * **XGBoost (XGBClassifier)**
  * **Máquinas de Soporte Vectorial (SVM)** con kernel RBF.

### 2. Aumento de Datos (*Data Augmentation*)
Para mitigar el sobreajuste provocado por la gesticulación diaria de los brazos y mejorar la generalización del modelo, el conjunto de entrenamiento incluye registros de sensores colocados en la cintura de forma dosificada:
* **Señales de Aceleración**: Incorporan un 30% de registros de cintura para regularizar la CNN.
* **Señales de Presión Barométrica**: Incorporan el 100% de los datos de cintura, ya que la variación neta de altura es análoga en ambas posiciones corporales.

### 3. Fusión de Inferencias
Se diseñaron y evaluaron dos estrategias de ensamble a nivel de decisión:
* **Fusión Secuencial (Cascada)**: El clasificador de acelerometría actúa como filtro primario de alta sensibilidad (umbral bajo). Si este sospecha de una caída, el clasificador barométrico evalúa el evento con un umbral más estricto para validar si existió un cambio neto de altura real, descartando así los movimientos parásitos de las manos.
* **Promedio Ponderado**: Combina linealmente las probabilidades de salida de ambos modelos ($P_{final} = k \cdot P_{baro} + (1-k) \cdot P_{accel}$) y evalúa el resultado contra un umbral de decisión global.

### Resultados de Desempeño
La arquitectura de fusión por **promedio ponderado** utilizando la **CNN (acelerómetro)** y el clasificador **XGBoost (barómetro)** obtuvo el mejor rendimiento:
* **F1-Score global**: 0.98
* **Sensibilidad (detección de caídas reales)**: 99%
* **Especificidad (rechazo de falsas alarmas)**: 95%
* **Comparativa**: Supera al modelo de referencia del estado del arte (*Saleh et al.*), el cual reporta un 87% de sensibilidad y un 94.5% de especificidad en dispositivos de muñeca.

---

## Estructura de Archivos del Proyecto

La organización del espacio de trabajo es la siguiente:

```text

├── models.py                             # Clases de Keras para los modelos Autoencoder y NN.
├── utils.py                              # Funciones auxiliares de preprocesamiento, métricas y graficación.
├── Autoencoder.ipynb                     # Notebook: Entrenamiento del autoencoder auto-supervisado.
├── CNN.ipynb                             # Notebook: Entrenamiento y pruebas de la CNN de acelerometría.
├── sensor_presion.ipynb                  # Notebook: Calibración barométrica y entrenamiento de clasificadores (NN, XGBoost, SVM).
├── baseline_model.ipynb                  # Notebook: Fusión de modelos (secuencial y ponderado) e inferencia integrada.
├── Coeficientes/                         # Directorio para guardar coeficientes de capas en formato CSV.
│   ├── coeficientes_encoder.csv          # Filtros y sesgos del codificador.
│   └── coeficientes_decoder.csv          # Filtros y sesgos del decodificador.
├── Datasets/                             # Enlaces/datos locales del directorio de código.
├── Modelos/                              # Copia local de archivos de modelos serializados.
├── Notebooks_tf/                         # Versiones alternativas o previas de los cuadernos de trabajo.
├── Papers/                               # Literatura y artículos científicos de referencia.
|   ├── Deteccion de caidas -Guido Glorioso.pdf   # Artículo científico/Paper técnico del proyecto.
└── imgs/                                 # Gráficas y esquemas exportados de los entrenamientos.
```

---

## Contenido de los Archivos más Relevantes

### 1. Scripts de Soporte
* **`utils.py`**: Centraliza el procesamiento de datos y la visualización del sistema. Incluye funciones para extraer muestras y etiquetas de los archivos de texto (`extraer_datos_csv`, `extraer_tags_csv`), escalar señales tridimensionales aplicando `StandardScaler` por canal (`dataScaler`), implementar el filtro de mediana móvil (`median_filter`), convertir la presión y temperatura en altitud (`calcular_altura`), y extraer las 12 características estadísticas de la señal barométrica (`extraer_features`). También contiene rutinas de graficación para curvas ROC, distribución de densidades y matrices de confusión detalladas por tipo de actividad.
* **`models.py`**: Define la arquitectura de los modelos de TensorFlow/Keras. Contiene la clase `Autoencoder` (compuesta por un *encoder* convolucional con capas `Conv2D` y `MaxPooling2D`, y un *decoder* con capas `Conv2DTranspose` y `UpSampling2D`) y la clase `NN` (la cabeza densa de clasificación con regularización L2 y capas de *Dropout*). Incorpora el método `set_coef` para inyectar coeficientes guardados en archivos CSV directamente en las capas del modelo, facilitando la portabilidad de los pesos sin depender únicamente de formatos binarios propietarios.

### 2. Cuadernos de Trabajo (Notebooks de Desarrollo Propio)
* **`Autoencoder.ipynb`**:
  * **Objetivo**: Entrenar un autoencoder de aprendizaje no supervisado para reconstruir las señales inerciales de aceleración de 3 ejes (forma de entrada: `[520, 3, 1]`).
  * **Proceso**: Carga los datos de acelerometría del dataset de muñeca, instancia la clase `Autoencoder` definida en `models.py` y la entrena minimizando el error cuadrático medio (`MeanSquaredError`). Al finalizar, se exportan los coeficientes del *encoder* y *decoder* en formato CSV a la carpeta `Coeficientes/` para su reutilización.
* **`CNN.ipynb`**:
  * **Objetivo**: Desarrollar, entrenar y evaluar la Red Neuronal Convolucional (CNN) supervisada para clasificar caídas a partir de acelerometría.
  * **Proceso**: Carga los coeficientes guardados del codificador para emplearlo como extractor de características inerciales fijas. Acopla la cabeza densa clasificadora (`NN`) e implementa el entrenamiento supervisado. Evalúa el desempeño cruzado del modelo de muñeca al ser validado con datos de cintura y aplica aumento de datos (mezcla del 30% de datos de cintura en el conjunto de entrenamiento) para mejorar la generalización. Finalmente, implementa funciones de rotación espacial de los datos para analizar el impacto de la desorientación física del sensor de muñeca.
* **`sensor_presion.ipynb`**:
  * **Objetivo**: Procesar los datos barométricos y entrenar los clasificadores encargados de estimar variaciones netas de altitud.
  * **Proceso**: Carga los datos de presión y temperatura tanto de la muñeca como de la cintura (incorporando el 100% para expandir la robustez estadística). Convierte las variables físicas en altitud relativa y extrae las 12 variables estadísticas tabulares mediante `extraer_features`. Con esta matriz de características, entrena y evalúa de forma comparativa tres modelos supervisados: una Red Neuronal densa feedforward (Keras), un clasificador XGBoost (`XGBClassifier`) optimizado mediante búsqueda heurística (*greedy search*), y una máquina de soporte vectorial (`SVC`) ajustada con pesos de clase específicos para priorizar la sensibilidad.
* **`baseline_model.ipynb`**:
  * **Objetivo**: Integrar las señales de acelerometría y presión barométrica y ejecutar las estrategias de fusión de inferencias.
  * **Proceso**: Carga los modelos previamente entrenados para ambos canales físicos. Implementa las funciones correspondientes a los esquemas de combinación **Secuencial (Cascada)** y **Promedio Ponderado**. Realiza una validación completa y exhaustiva sobre el conjunto de testeo de muñeca, evaluando diferentes combinaciones de modelos (CNN + NN, CNN + XGBoost, CNN + SVM) y sintonizando los umbrales de decisión mediante el análisis de curvas ROC. Genera las métricas de rendimiento consolidadas y expone gráficamente la distribución de errores (falsos positivos y falsos negativos) desglosados por tipo de actividad de la vida diaria (ADL) y tipo de caída.

---

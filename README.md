# Human Activity Recognition

Informe: [Project report](informe.pdf)

# Human Activity Recognition using Time-Series Feature Extraction

## Tabla de Contenidos
- [Resumen](#resumen)
- [Dataset](#dataset)
- [Metodología](#metodología)
  - [Extracción de Features](#extracción-de-features)
  - [Reducción de Dimensionalidad](#reducción-de-dimensionalidad)
- [Aspectos Clave de Implementación](#aspectos-clave-de-implementación)
- [Resultados](#resultados)

## Resumen

Este proyecto desarrolla un sistema de clasificación de actividades humanas a partir de datos de sensores de aceleración y giroscopios obtenidos desde un smartphone.

## Dataset

El dataset utilizado corresponde al problema clásico de **Human Activity Recognition (HAR)**.

El experimento original fue realizado con **30 voluntarios** que ejecutaron seis actividades distintas mientras llevaban un smartphone que registraba datos de sensores. Las actividades consideradas son:

- Caminar
- Subir escaleras
- Bajar escaleras
- Sentarse
- Estar de pie
- Estar acostado

Los sensores del teléfono registran información como:

- aceleración en los ejes **x, y, z**
- aceleración corporal (sin gravedad)
- velocidad angular del giroscopio

Cada observación corresponde a **series temporales de 128 puntos** por sensor.

El conjunto de datos está dividido en:

- **70 % entrenamiento**
- **30 % prueba**

Además, las clases presentan una distribución relativamente balanceada, lo cual permite utilizar **accuracy** como métrica principal de evaluación.

## Metodología

El pipeline del proyecto se compone de tres etapas principales:

1. Extracción de características de las series temporales  
2. Reducción de dimensionalidad  
3. Entrenamiento y evaluación de modelos de clasificación  

### Extracción de Features

Dado que los datos originales son **series temporales**, fue necesario transformarlos en vectores de características que puedan ser utilizados por modelos de machine learning.

Se experimentó con dos bibliotecas especializadas:

#### PyTS – ROCKET

Se utilizó el transformador **ROCKET (Random Convolutional KErnel Transform)** para generar representaciones de las series temporales mediante **kernels convolucionales aleatorios**.

Cada kernel aplica operaciones de convolución sobre la señal para capturar patrones relevantes en distintas escalas temporales. Este método destaca por:

- alta **eficiencia computacional**
- capacidad de generar **gran cantidad de features**
- buen desempeño en tareas de clasificación de series temporales

El número de características generadas depende del hiperparámetro **`n_kernels`**, donde cada kernel produce dos features.

#### TSFresh

También se experimentó con la librería **TSFresh**, que automatiza la extracción de características estadísticas de series temporales.

Entre los *features* que puede generar se encuentran:

- energía absoluta de la señal
- entropía aproximada
- número de picos
- entropía muestral
- coeficientes de wavelets

Además, TSFresh incluye funciones para **seleccionar automáticamente las características más relevantes** en función de la variable objetivo.

### Reducción de Dimensionalidad

Debido a la gran cantidad de features generadas, se aplicó **Principal Component Analysis (PCA)** para reducir la dimensionalidad del dataset.

PCA transforma el conjunto de variables originales en un nuevo espacio de menor dimensión compuesto por **componentes principales** que maximizan la varianza de los datos.

Este procedimiento permite:

- reducir el **costo computacional**
- disminuir el riesgo de **overfitting**
- conservar la mayor cantidad posible de información relevante

Los datos transformados se obtienen proyectando el dataset original sobre los **k eigenvectores principales** derivados de la matriz de covarianza.

PCA fue elegido principalmente por:

- su **eficiencia computacional**
- buenos resultados en las pruebas experimentales
- su efecto de **regularización implícita**

## Aspectos Clave de Implementación

La implementación del proyecto está organizada de forma modular.

Algunos aspectos relevantes incluyen:

- Separación entre **notebooks de experimentación** y **módulos de modelos**.
- Uso de **semillas aleatorias fijas** para garantizar replicabilidad (`random_state = 42` y `np.random.seed(2024)`).
- Experimentación con diferentes valores del hiperparámetro **`n_kernels`** de ROCKET para analizar el compromiso entre precisión y tiempo de entrenamiento.

Durante las pruebas se observó que:

- aumentar el número de kernels mejora ligeramente la precisión
- el costo computacional crece significativamente a partir de ciertos valores

Por esta razón se seleccionó **500 kernels**, generando **1000 features**, como un punto de equilibrio entre rendimiento y eficiencia.

## Resultados

Los experimentos mostraron diferencias importantes entre los métodos de extracción de características.

### Comparación de métodos de extracción

| Método | #Features | Tiempo de extracción |
|------|------|------|
| ROCKET (PyTS) | 1000 | ~37 s |
| TSFresh | 7047 | ~47 min |

ROCKET resultó **mucho más eficiente computacionalmente**, permitiendo generar grandes cantidades de features en tiempos significativamente menores.

### Rendimiento de los modelos

Utilizando PCA con 10 componentes principales, se obtuvieron los siguientes resultados aproximados de accuracy:

| Modelo | ROCKET | TSFresh |
|------|------|------|
| Regresión Logística | 84.37 % | 71.4 % |
| Árbol de Decisión | 79.10 % | 52.35 % |
| SVM | 47.99 % | 22.64 % |

Los resultados indican que:

- **ROCKET supera ampliamente a TSFresh** tanto en velocidad como en rendimiento.
- La **regresión logística** presenta el mejor balance entre precisión y eficiencia.
- SVM mostró el peor desempeño en este dataset.

### Observaciones del modelo

A partir de las matrices de confusión se identificaron patrones de error interesantes:

- Las actividades **SITTING** y **STANDING** presentan mayor confusión debido a la similitud en los patrones de movimiento.
- También se observaron errores ocasionales entre **subir** y **bajar escaleras**, probablemente por similitudes en los patrones de aceleración.

En general, los modelos lograron **altos niveles de precisión**, demostrando que la combinación de:

- extracción de features con **ROCKET**
- reducción de dimensionalidad con **PCA**

es efectiva para problemas de **clasificación de actividades humanas basadas en sensores**.

## Archivos adicioneles

Se creó una carpeta de drive en donde se guardó el dataset con el cuál se trabajó y los datos procesados por las librerías de extracción de features:
[Carpeta de Drive con datasets y datos procesados](https://drive.google.com/drive/folders/17wKPQCIJ6K-EzjWvMsl7nO84Wr_NpiQm?usp=sharing)

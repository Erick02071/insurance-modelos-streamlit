# Practica de modelado con datos de seguros

Practica de ciencia de datos que entrena, evalua, guarda y prueba modelos de
clasificacion, regresion y clustering con el conjunto `insurance.csv`. Todos
los modelos incluyen validacion cruzada de cinco particiones y una prueba con
datos no utilizados durante el entrenamiento.

## Dataset y variables objetivo

El dataset contiene 1,338 registros y 7 variables: `age`, `sex`, `bmi`,
`children`, `smoker`, `region` y `charges`. No contiene valores nulos y se
elimina un registro duplicado antes del modelado.

- Clasificacion: la variable objetivo es `level` (bajo, medio o alto), creada
  con K-means a partir de `charges`. Los clasificadores excluyen `charges` para
  evitar fuga de informacion.
- Regresion: la variable objetivo numerica es `charges`.
- Clustering: K-means segmenta `charges` en tres grupos.

## Modelos incluidos

| Tipo | Programa | Modelo |
|---|---|---|
| Clustering | `01_kmeans.py` | K-means |
| Clasificacion | `02_regresion_logistica.py` | Regresion logistica |
| Clasificacion | `03_knn.py` | KNN |
| Clasificacion | `04_svm.py` | SVM |
| Clasificacion | `05_naive_bayes.py` | Gaussian Naive Bayes |
| Regresion | `06_regresion_lineal.py` | Regresion lineal |
| Regresion | `07_arbol_decision.py` | Arbol de decision |
| Regresion | `08_random_forest.py` | Random Forest |

Cada programa tiene un archivo de prueba que carga el pipeline guardado con
`joblib` y realiza predicciones con casos sinteticos.

## Resultados principales

| Clasificador | Accuracy CV | Accuracy prueba | F1 macro |
|---|---:|---:|---:|
| Regresion logistica | 80.08 % | 83.58 % | 80.44 % |
| KNN | 83.35 % | 84.70 % | 80.34 % |
| SVM | **87.19 %** | **88.81 %** | **86.73 %** |
| Naive Bayes | 79.42 % | 79.85 % | 64.71 % |

| Regresor | R2 CV | R2 prueba | MAE | RMSE |
|---|---:|---:|---:|---:|
| Regresion lineal | 72.28 % | 80.69 % | $4,177.05 | $5,956.34 |
| Arbol de decision | 83.11 % | 89.55 % | $2,585.56 | $4,381.55 |
| Random Forest | **83.91 %** | **89.76 %** | **$2,394.21** | **$4,337.55** |

SVM obtuvo el mejor resultado de clasificacion y Random Forest obtuvo el mejor
resultado de regresion. La comparacion debe interpretarse dentro de cada tipo
de problema porque los clasificadores predicen niveles y los regresores
predicen un costo continuo.

## Estructura

- `data/`: dataset original.
- `modelos/`: pipelines y LabelEncoder guardados con `joblib`.
- `resultados/`: metricas, validacion cruzada, reportes, graficas y pruebas.
- `pages/`: paginas de la aplicacion Streamlit.
- `src/`: funciones compartidas por la aplicacion.
- `REPORTE_PRACTICA.md` y `REPORTE_PRACTICA_MODELADO.docx`: evidencia escrita.

## Instalacion

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Orden de ejecucion

```powershell
.\.venv\Scripts\python.exe 01_kmeans.py
.\.venv\Scripts\python.exe 02_regresion_logistica.py
.\.venv\Scripts\python.exe 03_knn.py
.\.venv\Scripts\python.exe 04_svm.py
.\.venv\Scripts\python.exe 05_naive_bayes.py
.\.venv\Scripts\python.exe 06_regresion_lineal.py
.\.venv\Scripts\python.exe 07_arbol_decision.py
.\.venv\Scripts\python.exe 08_random_forest.py
```

Despues se pueden ejecutar los ocho programas cuyo nombre contiene `prueba`.

## Aplicacion Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

La aplicacion permite explorar los datos, probar los modelos, segmentar costos
con K-means y comparar las metricas. Su uso es academico y sus predicciones no
representan una cotizacion real.

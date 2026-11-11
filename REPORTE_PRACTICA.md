# Reporte de practica de modelado con datos de seguros

**Estudiante:** ______________________________  
**Matricula:** __________________  
**Grupo:** __________________  
**Fecha:** 11 de noviembre de 2026

## Objetivo y conjunto de datos

La practica compara modelos de clasificacion y regresion aplicados al dataset
`insurance.csv`. El conjunto original tiene 1,338 filas y 7 columnas. Se
detecta un registro duplicado, que se elimina antes del entrenamiento, y no se
encuentran valores nulos. Las variables numericas son `age`, `bmi`, `children`
y `charges`; las variables categoricas son `sex`, `smoker` y `region`.

La edad promedio es 39.21 anos, el IMC promedio es 30.66 y el costo medico
promedio es $13,270.42. Los costos van de $1,121.87 a $63,770.43. El dataset
contiene 1,064 personas no fumadoras y 274 fumadoras.

## Preparacion y division

Se utiliza 80 % de los registros para entrenamiento y 20 % para prueba. La
clasificacion emplea division estratificada para conservar la proporcion de
los niveles bajo, medio y alto. Ademas, todos los modelos se evaluan mediante
validacion cruzada de cinco particiones. Esta combinacion permite medir el
rendimiento en datos no vistos y reduce el riesgo de elegir un modelo por el
resultado favorable de una sola division.

K-means crea la variable `level` a partir de `charges`. Para evitar fuga de
informacion, `charges` se excluye de los clasificadores. La variable `level` se
codifica con LabelEncoder. En regresion, `charges` es numerica y se utiliza
directamente como variable objetivo.

## Modelos y evaluacion

Los clasificadores entrenados son regresion logistica, KNN, SVM y Gaussian
Naive Bayes. Se evaluan con accuracy, balanced accuracy, precision, recall, F1
macro, reporte de clasificacion y matriz de confusion. Los regresores son
regresion lineal, arbol de decision y Random Forest. Se evaluan con R2, MAE,
MSE y RMSE.

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

## Analisis de resultados

SVM es el clasificador con mejor desempeno: alcanza 88.81 % de accuracy en la
prueba independiente y 86.73 % de F1 macro. Naive Bayes obtiene menor F1 macro,
principalmente porque identifica con menor precision el nivel medio. En
regresion, Random Forest presenta el mayor R2 de prueba, 89.76 %, y los menores
errores MAE y RMSE. El arbol de decision obtiene un resultado cercano, mientras
que la regresion lineal pierde precision porque la relacion entre las variables
y el costo no es completamente lineal.

Los resultados de clasificacion y regresion no se comparan como si resolvieran
la misma salida. La clasificacion sirve cuando se necesita asignar un nivel de
costo y la regresion cuando se requiere estimar una cantidad monetaria. Por
esta razon, SVM es la mejor opcion para predecir `level` y Random Forest es la
mejor opcion para estimar `charges`.

## Guardado y pruebas

Todos los pipelines se guardan con `joblib`. El LabelEncoder de clasificacion y
el mapa de niveles de K-means tambien se conservan. Cada uno de los cuatro
clasificadores y los tres regresores cuenta con un programa independiente que
carga el modelo guardado y realiza predicciones sobre tres casos sinteticos.
La aplicacion Streamlit permite repetir estas pruebas desde una interfaz y
GitHub mantiene el historial de cambios del proyecto.

## Conclusion

La practica demuestra que la seleccion del modelo depende de la variable que se
quiere predecir. La validacion cruzada confirma que SVM y Random Forest mantienen
un desempeno superior a los modelos mas simples. Los modelos, codificadores,
metricas, graficas y pruebas quedan guardados para reproducir los resultados y
utilizarlos en la aplicacion Streamlit.

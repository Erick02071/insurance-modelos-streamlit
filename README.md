# Insurance Modelos Streamlit

Aplicacion multipagina para explorar el dataset de seguros y probar modelos de
clasificacion, regresion y clustering.

## Modelos

- Clasificacion: regresion logistica, KNN, SVM y Naive Bayes.
- Regresion: regresion lineal, arbol de decision y Random Forest.
- Clustering: K-means con tres niveles de costo.
- Todos los modelos fueron evaluados mediante validacion cruzada de cinco
  particiones.

## Ejecucion local

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

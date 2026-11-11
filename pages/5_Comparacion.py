import pandas as pd
import streamlit as st

from src.utils import cargar_metricas, configurar_pagina


configurar_pagina("Comparacion de modelos", "📈")
st.title("📈 Comparacion de modelos")
st.write("Metricas obtenidas con la misma division de entrenamiento y prueba.")

clasificadores = {
    "Regresion logistica": "metricas_regresion_logistica.json",
    "KNN": "metricas_knn.json",
    "SVM": "metricas_svm.json",
    "Naive Bayes": "metricas_naive_bayes.json",
}
filas_clasificacion = []
for modelo, archivo in clasificadores.items():
    m = cargar_metricas(archivo)
    filas_clasificacion.append(
        {
            "Modelo": modelo,
            "Accuracy CV (%)": m.get("cv_accuracy_promedio", 0) * 100,
            "Accuracy prueba (%)": m.get("test_accuracy", 0) * 100,
            "Balanced accuracy (%)": m.get("test_balanced_accuracy", 0) * 100,
            "F1 macro (%)": m.get("test_f1_macro", 0) * 100,
        }
    )
tabla_clasificacion = pd.DataFrame(filas_clasificacion).set_index("Modelo")

st.subheader("Modelos de clasificacion")
st.dataframe(tabla_clasificacion.style.format("{:.2f}"), width="stretch")
st.bar_chart(
    tabla_clasificacion[["Accuracy prueba (%)", "F1 macro (%)"]],
    stack=False,
)

regresores = {
    "Regresion lineal": "metricas_regresion_lineal.json",
    "Arbol de decision": "metricas_arbol_decision.json",
    "Random Forest": "metricas_random_forest.json",
}
filas_regresion = []
for modelo, archivo in regresores.items():
    m = cargar_metricas(archivo)
    filas_regresion.append(
        {
            "Modelo": modelo,
            "R² CV (%)": m.get("cv_r2_promedio", 0) * 100,
            "R² prueba (%)": m.get("test_r2", 0) * 100,
            "MAE": m.get("test_mae", 0),
            "RMSE": m.get("test_rmse", 0),
        }
    )
tabla_regresion = pd.DataFrame(filas_regresion).set_index("Modelo")

st.subheader("Modelos de regresion")
st.dataframe(
    tabla_regresion.style.format(
        {
            "R² CV (%)": "{:.2f}",
            "R² prueba (%)": "{:.2f}",
            "MAE": "${:,.2f}",
            "RMSE": "${:,.2f}",
        }
    ),
    width="stretch",
)
st.bar_chart(
    tabla_regresion[["R² CV (%)", "R² prueba (%)"]],
    stack=False,
)

mejor_clasificador = tabla_clasificacion["Accuracy prueba (%)"].idxmax()
mejor_regresor = tabla_regresion["R² prueba (%)"].idxmax()
c1, c2 = st.columns(2)
c1.success(f"Mejor accuracy de prueba: **{mejor_clasificador}**")
c2.success(f"Mejor R² de prueba: **{mejor_regresor}**")

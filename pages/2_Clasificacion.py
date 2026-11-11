import pandas as pd
import streamlit as st

from src.utils import (
    cargar_metricas,
    cargar_modelo,
    configurar_pagina,
    formulario_persona,
)


configurar_pagina("Clasificacion", "🎯")
st.title("🎯 Clasificacion del nivel de costo")
st.write(
    "Predice si una persona pertenece al nivel de costo bajo, medio o alto. "
    "La variable `charges` no se utiliza como entrada para evitar fuga de informacion."
)

opciones = {
    "Regresion logistica": (
        "pipeline_regresion_logistica.joblib",
        "metricas_regresion_logistica.json",
    ),
    "KNN": ("pipeline_knn.joblib", "metricas_knn.json"),
    "SVM": ("pipeline_svm.joblib", "metricas_svm.json"),
    "Naive Bayes": ("pipeline_naive_bayes.joblib", "metricas_naive_bayes.json"),
}
nombre = st.selectbox("Modelo de clasificacion", list(opciones))
archivo_modelo, archivo_metricas = opciones[nombre]
metricas = cargar_metricas(archivo_metricas)

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy de prueba", f"{metricas.get('test_accuracy', 0) * 100:.2f} %")
m2.metric(
    "Balanced accuracy",
    f"{metricas.get('test_balanced_accuracy', 0) * 100:.2f} %",
)
m3.metric("F1 macro", f"{metricas.get('test_f1_macro', 0) * 100:.2f} %")

st.subheader("Datos del nuevo caso")
caso = formulario_persona("clasificacion")

if st.button("Predecir nivel", type="primary", width="stretch"):
    modelo = cargar_modelo(archivo_modelo)
    codificador = cargar_modelo("label_encoder_clasificacion.joblib")
    codigo = modelo.predict(caso)
    nivel = codificador.inverse_transform(codigo.astype(int))[0]
    st.success(f"Nivel predicho: **{nivel.upper()}**")

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(caso)[0]
        etiquetas = codificador.inverse_transform(modelo.classes_.astype(int))
        tabla = pd.DataFrame(
            {"Nivel": etiquetas, "Probabilidad": probabilidades}
        ).set_index("Nivel")
        st.bar_chart(tabla)
        st.dataframe(
            (tabla * 100).rename(columns={"Probabilidad": "Probabilidad (%)"}),
            width="stretch",
        )

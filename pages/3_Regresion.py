import streamlit as st

from src.utils import (
    cargar_metricas,
    cargar_modelo,
    configurar_pagina,
    formulario_persona,
)


configurar_pagina("Regresion", "💵")
st.title("💵 Prediccion del costo del seguro")
st.write("Estima directamente el valor de `charges` para un nuevo caso.")

opciones = {
    "Regresion lineal": (
        "pipeline_regresion_lineal.joblib",
        "metricas_regresion_lineal.json",
    ),
    "Arbol de decision": (
        "pipeline_arbol_decision.joblib",
        "metricas_arbol_decision.json",
    ),
    "Random Forest": (
        "pipeline_random_forest.joblib",
        "metricas_random_forest.json",
    ),
}
nombre = st.selectbox("Modelo de regresion", list(opciones))
archivo_modelo, archivo_metricas = opciones[nombre]
metricas = cargar_metricas(archivo_metricas)

m1, m2, m3 = st.columns(3)
m1.metric("R² de prueba", f"{metricas.get('test_r2', 0) * 100:.2f} %")
m2.metric("MAE", f"${metricas.get('test_mae', 0):,.2f}")
m3.metric("RMSE", f"${metricas.get('test_rmse', 0):,.2f}")

st.subheader("Datos del nuevo caso")
caso = formulario_persona("regresion")

if st.button("Estimar costo", type="primary", width="stretch"):
    modelo = cargar_modelo(archivo_modelo)
    costo = float(modelo.predict(caso)[0])
    st.success(f"Costo medico estimado: **${max(costo, 0):,.2f}**")
    st.caption(f"Prediccion generada mediante {nombre}.")

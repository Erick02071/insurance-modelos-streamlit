import pandas as pd
import streamlit as st

from src.utils import (
    cargar_datos_niveles,
    cargar_modelo,
    configurar_pagina,
)


configurar_pagina("K-means", "🧩")
datos = cargar_datos_niveles()

st.title("🧩 Segmentacion de costos con K-means")
st.write(
    "K-means agrupa los costos medicos en tres segmentos y les asigna las "
    "etiquetas bajo, medio y alto segun el orden de sus centroides."
)

resumen = (
    datos.groupby("level")["charges"]
    .agg(registros="count", promedio="mean", minimo="min", maximo="max")
    .reindex(["bajo", "medio", "alto"])
)
st.dataframe(
    resumen.style.format(
        {"promedio": "${:,.2f}", "minimo": "${:,.2f}", "maximo": "${:,.2f}"}
    ),
    width="stretch",
)

st.subheader("Clasificar un costo nuevo")
costo = st.number_input(
    "Costo medico (charges)", min_value=0.0, max_value=100000.0, value=12000.0, step=500.0
)
if st.button("Asignar nivel", type="primary", width="stretch"):
    modelo = cargar_modelo("pipeline_kmeans.joblib")
    mapa = cargar_modelo("mapa_niveles_kmeans.joblib")
    cluster = int(modelo.predict(pd.DataFrame({"charges": [costo]}))[0])
    nivel = mapa[cluster]
    st.success(f"El costo ${costo:,.2f} pertenece al nivel **{nivel.upper()}**.")

st.subheader("Cantidad de registros por nivel")
st.bar_chart(resumen[["registros"]])

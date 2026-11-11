import streamlit as st

from src.utils import cargar_datos, configurar_pagina


configurar_pagina("Datos de seguros", "📊")
datos = cargar_datos()

st.title("📊 Datos de seguros")
st.write("Explora los registros utilizados para entrenar los modelos.")

col1, col2, col3 = st.columns(3)
with col1:
    fumador = st.multiselect("Fumador", sorted(datos["smoker"].unique()))
with col2:
    regiones = st.multiselect("Region", sorted(datos["region"].unique()))
with col3:
    rango_edad = st.slider(
        "Rango de edad",
        int(datos["age"].min()),
        int(datos["age"].max()),
        (int(datos["age"].min()), int(datos["age"].max())),
    )

filtrados = datos[datos["age"].between(*rango_edad)].copy()
if fumador:
    filtrados = filtrados[filtrados["smoker"].isin(fumador)]
if regiones:
    filtrados = filtrados[filtrados["region"].isin(regiones)]

c1, c2, c3 = st.columns(3)
c1.metric("Registros mostrados", f"{len(filtrados):,}")
c2.metric("Costo promedio", f"${filtrados['charges'].mean():,.2f}")
c3.metric("IMC promedio", f"{filtrados['bmi'].mean():.2f}")

st.dataframe(filtrados, width="stretch", hide_index=True)
st.subheader("Distribucion de costos")
st.bar_chart(filtrados["charges"].value_counts(bins=20).sort_index())

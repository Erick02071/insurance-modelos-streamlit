import streamlit as st

from src.utils import cargar_datos, configurar_pagina


configurar_pagina("Modelos de seguros", "🛡️")
datos = cargar_datos()

st.title("🛡️ Modelos predictivos de seguros")
st.write(
    "Aplicacion interactiva para explorar el dataset de seguros y probar "
    "modelos de clasificacion, regresion y clustering."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros", f"{len(datos):,}")
col2.metric("Variables", len(datos.columns))
col3.metric("Modelos supervisados", 7)
col4.metric("Validacion cruzada", "5 folds")

st.subheader("¿Que contiene la aplicacion?")
st.markdown(
    """
    - **Datos:** consulta y filtros del conjunto `insurance.csv`.
    - **Clasificacion:** predice niveles de costo con cuatro algoritmos.
    - **Regresion:** estima directamente el costo medico `charges`.
    - **K-means:** asigna costos a los niveles bajo, medio o alto.
    - **Comparacion:** presenta las metricas de validacion y prueba.
    """
)

st.info(
    "Los resultados son demostrativos y academicos. No representan una "
    "cotizacion real ni deben utilizarse para decisiones medicas o financieras."
)

st.subheader("Variables utilizadas")
st.dataframe(
    {
        "Variable": ["age", "sex", "bmi", "children", "smoker", "region", "charges"],
        "Descripcion": [
            "Edad de la persona",
            "Sexo registrado",
            "Indice de masa corporal",
            "Numero de hijos",
            "Condicion de fumador",
            "Region de residencia",
            "Costo medico del seguro",
        ],
    },
    width="stretch",
    hide_index=True,
)

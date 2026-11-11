import streamlit as st


st.set_page_config(
    page_title="Modelos de seguros",
    page_icon="🛡️",
    layout="wide",
)

paginas = [
    st.Page("inicio.py", title="Inicio", icon="🏠", default=True),
    st.Page("pages/1_Datos.py", title="Datos", icon="📊"),
    st.Page("pages/2_Clasificacion.py", title="Clasificacion", icon="🎯"),
    st.Page("pages/3_Regresion.py", title="Regresion", icon="💵"),
    st.Page("pages/4_Kmeans.py", title="K-means", icon="🧩"),
    st.Page("pages/5_Comparacion.py", title="Comparacion", icon="📈"),
]

navegacion = st.navigation(paginas)
navegacion.run()

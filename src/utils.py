from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELOS = BASE / "modelos"
RESULTADOS = BASE / "resultados"


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(DATA / "insurance.csv")


@st.cache_data
def cargar_datos_niveles() -> pd.DataFrame:
    return pd.read_csv(RESULTADOS / "insurance_with_levels.csv")


@st.cache_resource
def cargar_modelo(nombre: str):
    ruta = MODELOS / nombre
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontro el modelo: {ruta.name}")
    return joblib.load(ruta)


@st.cache_data
def cargar_metricas(nombre: str) -> dict:
    ruta = RESULTADOS / nombre
    if not ruta.is_file():
        return {}
    return json.loads(ruta.read_text(encoding="utf-8"))


def formulario_persona(clave: str) -> pd.DataFrame:
    col1, col2, col3 = st.columns(3)
    with col1:
        edad = st.number_input("Edad", 18, 100, 35, key=f"edad_{clave}")
        sexo = st.selectbox("Sexo", ["female", "male"], key=f"sexo_{clave}")
    with col2:
        bmi = st.number_input(
            "IMC", 10.0, 60.0, 27.5, 0.1, key=f"bmi_{clave}"
        )
        fumador = st.selectbox(
            "Fumador", ["no", "yes"], key=f"fumador_{clave}"
        )
    with col3:
        hijos = st.number_input("Hijos", 0, 10, 0, key=f"hijos_{clave}")
        region = st.selectbox(
            "Region",
            ["northeast", "northwest", "southeast", "southwest"],
            key=f"region_{clave}",
        )

    return pd.DataFrame(
        [
            {
                "age": edad,
                "bmi": bmi,
                "children": hijos,
                "sex": sexo,
                "smoker": fumador,
                "region": region,
            }
        ]
    )


def configurar_pagina(titulo: str, icono: str = "🛡️") -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {
            background: #f4f7fb;
            border: 1px solid #d9e2ef;
            border-radius: 12px;
            padding: 14px;
        }
        div[data-testid="stAlert"] {border-radius: 12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

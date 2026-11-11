# -*- coding: utf-8 -*-
"""Carga el pipeline SVM guardado y predice casos nuevos."""

from pathlib import Path

import joblib
import pandas as pd


BASE = Path(__file__).resolve().parent
ARCHIVO_MODELO = BASE / "modelos" / "pipeline_svm.joblib"
ARCHIVO_CODIFICADOR = BASE / "modelos" / "label_encoder_clasificacion.joblib"
ARCHIVO_SALIDA = BASE / "resultados" / "prueba_svm.csv"

VARIABLES = ["age", "bmi", "children", "sex", "smoker", "region"]


def ejecutar_prueba() -> None:
    if not ARCHIVO_MODELO.is_file() or not ARCHIVO_CODIFICADOR.is_file():
        raise FileNotFoundError("Faltan los activos. Ejecuta primero 04_svm.py.")

    modelo = joblib.load(ARCHIVO_MODELO)
    codificador = joblib.load(ARCHIVO_CODIFICADOR)

    casos = pd.DataFrame(
        [
            {
                "age": 24,
                "bmi": 23.1,
                "children": 0,
                "sex": "male",
                "smoker": "no",
                "region": "northwest",
            },
            {
                "age": 43,
                "bmi": 31.0,
                "children": 3,
                "sex": "female",
                "smoker": "no",
                "region": "northeast",
            },
            {
                "age": 55,
                "bmi": 37.2,
                "children": 2,
                "sex": "male",
                "smoker": "yes",
                "region": "southeast",
            },
        ],
        columns=VARIABLES,
    )

    predicciones_codificadas = modelo.predict(casos)
    probabilidades = modelo.predict_proba(casos)
    niveles = codificador.inverse_transform(predicciones_codificadas)

    resultados = casos.copy()
    resultados["nivel_predicho"] = niveles
    for indice, clase in enumerate(codificador.classes_):
        resultados[f"probabilidad_{clase}"] = probabilidades[:, indice]

    resultados.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")

    print("=== PRUEBA DEL MODELO SVM GUARDADO ===")
    for numero, fila in resultados.iterrows():
        print(f"\nCaso {numero + 1}: nivel {fila['nivel_predicho'].upper()}")
        for clase in codificador.classes_:
            print(f"  {clase}: {fila[f'probabilidad_{clase}'] * 100:.2f} %")
    print(f"\nResultados guardados en: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    ejecutar_prueba()

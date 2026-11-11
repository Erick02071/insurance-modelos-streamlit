# -*- coding: utf-8 -*-
"""Carga el arbol guardado y predice tres casos nuevos."""

from pathlib import Path

import joblib
import pandas as pd


BASE = Path(__file__).resolve().parent
RUTA_MODELO = BASE / "modelos" / "pipeline_arbol_decision.joblib"
RUTA_SALIDA = BASE / "resultados" / "prueba_arbol_decision.csv"


def ejecutar() -> None:
    if not RUTA_MODELO.is_file():
        raise FileNotFoundError(
            "No se encontro el modelo. Ejecuta primero 07_arbol_decision.py."
        )

    casos = pd.DataFrame(
        [
            {
                "age": 21,
                "bmi": 22.5,
                "children": 0,
                "sex": "female",
                "smoker": "no",
                "region": "southwest",
            },
            {
                "age": 45,
                "bmi": 30.2,
                "children": 2,
                "sex": "male",
                "smoker": "no",
                "region": "northwest",
            },
            {
                "age": 58,
                "bmi": 36.8,
                "children": 1,
                "sex": "female",
                "smoker": "yes",
                "region": "southeast",
            },
        ]
    )

    modelo = joblib.load(RUTA_MODELO)
    predicciones = modelo.predict(casos)
    salida = casos.copy()
    salida["charges_predicho"] = predicciones

    print("=== PRUEBA DEL ARBOL DE DECISION GUARDADO ===")
    for indice, costo in enumerate(predicciones, start=1):
        print(f"Caso {indice}: costo estimado ${costo:,.2f}")

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(RUTA_SALIDA, index=False, encoding="utf-8-sig")
    print(f"\nResultados guardados en: {RUTA_SALIDA}")


if __name__ == "__main__":
    ejecutar()
